import pandas as pd
from datetime import datetime, timezone
import mysql.connector
import time
import requests
import logging

from dotenv import load_dotenv
import os
import mysql.connector

load_dotenv()

conn_mysql = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME")
)


API_KEY_OPENAQ = os.getenv("API_KEY")

# --- Logger ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Fonctions utilitaires ---
def iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def safe_request(url, headers=None, params=None, max_retries=3, sleep_base=1.2):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            r.raise_for_status()
            time.sleep(sleep_base)
            return r
        except requests.exceptions.HTTPError as e:
            if r.status_code == 429:
                wait_time = 60 * (attempt + 1)
                logger.warning(f"Rate limit atteint, pause {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.warning(f"Erreur HTTP {r.status_code}, tentative {attempt + 1}: {e}")
                time.sleep(5)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Erreur de connexion, tentative {attempt + 1}: {e}")
            time.sleep(5)
    raise Exception(f"Échec de la requête après {max_retries} tentatives: {url}")

# --- ETL Stations & Capteurs ---
def fetch_stations(lat, lon, radius=10000, limit=100):
    url = "https://api.openaq.org/v3/locations"
    headers = {"X-API-Key": API_KEY_OPENAQ}
    params = {"coordinates": f"{lat},{lon}", "radius": radius, "limit": limit}
    r = safe_request(url, headers=headers, params=params)
    return r.json().get("results", [])

def transform_stations_and_sensors(stations: list):
    stations_tuples, sensors_tuples = [], []
    for st in stations:
        stations_tuples.append((st["id"], st.get("name"), st["coordinates"]["latitude"], st["coordinates"]["longitude"]))
        for sensor in st.get("sensors", []):
            sensors_tuples.append((sensor["id"], st["id"], sensor.get("parameter", {}).get("name"), sensor.get("parameter", {}).get("units")))
    return stations_tuples, sensors_tuples

# --- ETL Pollution ---
def fetch_pollution_hourly_from_api(sensor_id, datetime_from, datetime_to, limit=100):
    url = f"https://api.openaq.org/v3/sensors/{sensor_id}/hours"
    headers = {"X-API-Key": API_KEY_OPENAQ}
    all_results, page = [], 1
    while True:
        params = {
            "datetime_from": iso_utc(datetime_from),
            "datetime_to": iso_utc(datetime_to),
            "limit": limit,
            "page": page
        }
        logger.info(f"Requête page {page} pour capteur {sensor_id}")
        r = safe_request(url, headers=headers, params=params)
        results = r.json().get("results", [])
        if not results:
            break
        all_results.extend(results)
        if len(results) < limit:
            break
        page += 1
    return all_results

def transform_pollution_data(raw_data, sensor_id):
    tuples = []
    for res in raw_data:
        value = res.get("value")
        if value is None or value < 0 or res.get("hasFlags", False):
            continue
        dt_utc = (res.get("period", {}).get("datetimeFrom") or {}).get("utc")
        if dt_utc:
            tuples.append((sensor_id, pd.to_datetime(dt_utc, utc=True).to_pydatetime(), value))
    return tuples

# --- ETL Weather ---
def fetch_weather_bulk_from_api(lat, lon, start, end):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,cloudcover,windspeed_10m,pressure_msl,winddirection_10m,shortwave_radiation",
        "timezone": "UTC"
    }
    r = safe_request(url, params=params)
    res = r.json().get("hourly", {})
    if not res or "time" not in res:
        return pd.DataFrame()
    
    df_weather = pd.DataFrame(res).rename(columns={
        "relative_humidity_2m": "humidity",
        "windspeed_10m": "windspeed"
    })
    df_weather["datetime_utc"] = pd.to_datetime(df_weather["time"], utc=True)
    return df_weather.drop(columns=["time"])

def transform_weather_data(df_weather, station_id):
    df_clean = df_weather.dropna(subset=[
        "temperature_2m","humidity","precipitation","cloudcover","windspeed",
        "pressure_msl","winddirection_10m","shortwave_radiation"
    ])
    return [
        (
            station_id,
            row.datetime_utc.to_pydatetime().replace(tzinfo=None),
            row.temperature_2m,
            row.humidity,
            row.precipitation,
            row.cloudcover,
            row.windspeed,
            getattr(row, "pressure_msl", None),
            getattr(row, "winddirection_10m", None),
            getattr(row, "shortwave_radiation", None)
        )
        for row in df_clean.itertuples(index=False)
    ]

# --- Base de données ---
def load_data(cursor, conn, table, columns, data, batch_size=500):
    if not data:
        return 0
    placeholders = ",".join(["%s"] * len(columns))
    sql = f"INSERT IGNORE INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
    inserted = 0
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        cursor.executemany(sql, batch)
        conn.commit()
        inserted += cursor.rowcount
    return inserted

def setup_database(cursor):
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stations (
        station_id INT PRIMARY KEY,
        name VARCHAR(100),
        latitude DOUBLE NOT NULL,
        longitude DOUBLE NOT NULL
    )""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sensors (
        sensor_id INT PRIMARY KEY,
        station_id INT NOT NULL,
        parameter VARCHAR(50) NOT NULL,
        unit VARCHAR(10),
        FOREIGN KEY (station_id) REFERENCES stations(station_id)
    )""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pollution_measurements (
        sensor_id INT NOT NULL,
        datetime_utc DATETIME NOT NULL,
        value DOUBLE NOT NULL,
        PRIMARY KEY (sensor_id, datetime_utc),
        FOREIGN KEY (sensor_id) REFERENCES sensors(sensor_id)
    )""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS weather_measurements (
        station_id INT NOT NULL,
        datetime_utc DATETIME NOT NULL,
        temperature_2m DOUBLE,
        humidity DOUBLE,
        precipitation DOUBLE,
        cloudcover DOUBLE,
        windspeed DOUBLE,
        pressure_msl DOUBLE,
        winddirection_10m DOUBLE,
        shortwave_radiation DOUBLE,
        PRIMARY KEY (station_id, datetime_utc),
        FOREIGN KEY (station_id) REFERENCES stations(station_id)
    )
    """)

def get_last_pollution_date(cursor, sensor_id):
    cursor.execute("SELECT MAX(datetime_utc) FROM pollution_measurements WHERE sensor_id = %s", (sensor_id,))
    result = cursor.fetchone()[0]
    return result.replace(tzinfo=timezone.utc) if result else None

def get_last_weather_date(cursor, station_id):
    cursor.execute("SELECT MAX(datetime_utc) FROM weather_measurements WHERE station_id = %s", (station_id,))
    result = cursor.fetchone()[0]
    return result.replace(tzinfo=timezone.utc) if result else None

# --- Orchestration ---
def get_data(lat, lon, days=730):
    try:
        conn = mysql.connector.connect(    
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"))
        cursor = conn.cursor()
        logger.info("Connexion à la base de données réussie.")
        setup_database(cursor)

        stations_raw = fetch_stations(lat, lon)
        if not stations_raw:
            logger.warning("Aucune station trouvée, fin du processus.")
            return

        stations_tuples, sensors_tuples = transform_stations_and_sensors(stations_raw)
        logger.info(f"{load_data(cursor, conn, 'stations', ['station_id','name','latitude','longitude'], stations_tuples)} stations insérées.")
        logger.info(f"{load_data(cursor, conn, 'sensors', ['sensor_id','station_id','parameter','unit'], sensors_tuples)} capteurs insérés.")

        for st in stations_raw:
            for sensor in st.get("sensors", []):
                sensor_id = sensor["id"]
                dt_to = datetime.now(timezone.utc)
                dt_from = get_last_pollution_date(cursor, sensor_id) or dt_to - pd.Timedelta(days=days)
                if dt_from < dt_to:
                    raw_pollution = fetch_pollution_hourly_from_api(sensor_id, dt_from, dt_to)
                    tuples = transform_pollution_data(raw_pollution, sensor_id)
                    inserted = load_data(cursor, conn, "pollution_measurements", ["sensor_id","datetime_utc","value"], tuples)
                    logger.info(f"{inserted} mesures pollution insérées pour capteur {sensor_id}.")
                else:
                    logger.info(f"Données pollution déjà à jour pour capteur {sensor_id}.")

        for st in stations_raw:
            station_id = st["id"]
            dt_to = datetime.now(timezone.utc)
            dt_from = get_last_weather_date(cursor, station_id) or dt_to - pd.Timedelta(days=days)
            if dt_from < dt_to:
                df_weather = fetch_weather_bulk_from_api(st["coordinates"]["latitude"], st["coordinates"]["longitude"], dt_from, dt_to)
                if not df_weather.empty:
                    tuples = transform_weather_data(df_weather, station_id)
                    # Ici on prend bien toutes les 10 colonnes
                    inserted = load_data(cursor, conn, "weather_measurements",
                                         ["station_id","datetime_utc","temperature_2m","humidity","precipitation","cloudcover",
                                          "windspeed","pressure_msl","winddirection_10m","shortwave_radiation"], tuples)
                    logger.info(f"{inserted} mesures météo insérées pour station {station_id}.")
                else:
                    logger.warning(f"Aucune donnée météo valide pour station {station_id}.")
            else:
                logger.info(f"Données météo déjà à jour pour station {station_id}.")

    except Exception as e:
        logger.error(f"Erreur fatale: {e}", exc_info=True)
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()
        logger.info("Processus terminé.")

# Point d'entrée
if __name__ == "__main__":
    get_data(48.8566, 2.3522)
