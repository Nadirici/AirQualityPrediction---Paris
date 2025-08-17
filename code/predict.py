# predict_24h_plot_live.py

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import mysql.connector
from datetime import datetime, timedelta
import os

from dotenv import load_dotenv
import os
import mysql.connector

load_dotenv()


# -----------------------
# Paramètres
# -----------------------
seq_hours = 72
target_pollutants = ['no2','pm10','o3','pm25']

# -----------------------
# Chemins des modèles et scalers
# -----------------------
model_path = os.path.join(os.path.dirname(__file__), "model", "pollution_model")
feature_scaler_path = os.path.join(os.path.dirname(__file__), "model", "feature_scaler.pkl")
target_scaler_path = os.path.join(os.path.dirname(__file__), "model", "target_scaler.pkl")

print("🔌 Chargement du modèle et des scalers…")
model = tf.keras.models.load_model(model_path, compile=False)
feature_scaler = joblib.load(feature_scaler_path)
target_scaler = joblib.load(target_scaler_path)

# -----------------------
# Connexion MySQL et récupération des données
# -----------------------
print("🔌 Connexion à MySQL pour récupérer les dernières données…")
conn_mysql = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME")
)


query = f"""
SELECT pm.sensor_id, s.parameter, pm.datetime_utc, pm.value AS pollution_value,
       s.station_id, w.temperature_2m, w.humidity, w.precipitation, w.cloudcover, w.windspeed,
       w.pressure_msl, w.winddirection_10m, w.shortwave_radiation
FROM pollution_measurements pm
JOIN sensors s ON pm.sensor_id = s.sensor_id
JOIN weather_measurements w ON s.station_id = w.station_id AND pm.datetime_utc = w.datetime_utc
WHERE s.parameter IN ({','.join([f"'{p}'" for p in target_pollutants])})
ORDER BY pm.datetime_utc
"""
df = pd.read_sql(query, conn_mysql, parse_dates=['datetime_utc'])
conn_mysql.close()
print(f"✅ Données chargées : {df.shape[0]} lignes")

# -----------------------
# Préparation des données horaires
# -----------------------
df['hour'] = df['datetime_utc'].dt.floor('h')

df_hourly = df.groupby(['hour','parameter'], as_index=False)['pollution_value'].mean()\
              .pivot(index='hour', columns='parameter', values='pollution_value')
df_hourly = df_hourly.reindex(columns=target_pollutants)

weather_cols = ['temperature_2m','humidity','precipitation','cloudcover','windspeed',
                'pressure_msl','winddirection_10m','shortwave_radiation']
df_weather = df.groupby('hour', as_index=False)[weather_cols].mean().set_index('hour')
df_hourly = df_hourly.join(df_weather, how='left')

df_hourly = df_hourly.interpolate(method='time', limit_direction='both')
df_hourly = df_hourly.fillna(df_hourly.mean())

# Features temporelles
df_hourly['weekday'] = df_hourly.index.weekday
df_hourly['month'] = df_hourly.index.month
hours = df_hourly.index.hour
df_hourly['hour_sin'] = np.sin(2 * np.pi * hours / 24)
df_hourly['hour_cos'] = np.cos(2 * np.pi * hours / 24)

# Lags et deltas
lags = [1,2,3,24,48]
for pol in target_pollutants:
    for lag in lags:
        df_hourly[f'{pol}_lag{lag}'] = df_hourly[pol].shift(lag).bfill()
    df_hourly[f'{pol}_delta'] = df_hourly[pol] - df_hourly[pol].shift(1).bfill()

# Interactions
for meteo in weather_cols:
    for pol in target_pollutants:
        df_hourly[f'{meteo}_x_{pol}'] = df_hourly[meteo] * df_hourly[pol]
for time_feat in ['hour_sin', 'hour_cos']:
    for pol in target_pollutants:
        df_hourly[f'{time_feat}_x_{pol}'] = df_hourly[time_feat] * df_hourly[pol]
for i, pol1 in enumerate(target_pollutants):
    for pol2 in target_pollutants[i+1:]:
        df_hourly[f'{pol1}_x_{pol2}'] = df_hourly[pol1] * df_hourly[pol2]

df_hourly = df_hourly.sort_index()

# -----------------------
# Colonnes features
# -----------------------
feature_cols = weather_cols + ['weekday','month','hour_sin','hour_cos']
lag_cols = [col for col in df_hourly.columns if '_lag' in col or '_delta' in col]
interaction_cols = [col for col in df_hourly.columns if '_x_' in col]
feature_cols_all = feature_cols + lag_cols + interaction_cols

# -----------------------
# Scaling avec gestion des colonnes manquantes et ordre exact
# -----------------------
for col in feature_scaler.feature_names_in_:
    if col not in df_hourly.columns:
        df_hourly[col] = 0

df_hourly = df_hourly.reindex(columns=list(feature_scaler.feature_names_in_) + target_pollutants)
df_hourly[feature_scaler.feature_names_in_] = feature_scaler.transform(df_hourly[feature_scaler.feature_names_in_])
df_hourly[target_pollutants] = target_scaler.transform(df_hourly[target_pollutants])

# Construction de la dernière séquence
last_sequence = df_hourly[list(feature_scaler.feature_names_in_) + target_pollutants].values[-seq_hours:]
print(f"✅ Dernière séquence créée : {last_sequence.shape}")

# -----------------------
# Fonction pour prédire les 24 prochaines heures
# -----------------------
def predict_next_24h(model, last_sequence, target_scaler):
    n_outputs = len(target_pollutants)
    predictions_scaled = []
    current_sequence = last_sequence.copy()
    
    for _ in range(24):
        X_input = current_sequence[np.newaxis, :, :]
        y_pred_scaled = model.predict(X_input, verbose=0)[0]
        predictions_scaled.append(y_pred_scaled)
        
        next_row = current_sequence[-1, :].copy()
        next_row[-n_outputs:] = y_pred_scaled
        current_sequence = np.vstack([current_sequence[1:], next_row])
    
    predictions_scaled = np.array(predictions_scaled, dtype=np.float32)
    predictions_denorm = target_scaler.inverse_transform(predictions_scaled)
    return predictions_denorm

# -----------------------
# Génération des prédictions
# -----------------------
pred_next_24h = predict_next_24h(model, last_sequence, target_scaler)

dates_future = pd.date_range(start=pd.Timestamp.now(), periods=24, freq='H')

fig, axes = plt.subplots(len(target_pollutants), 1, figsize=(14, 3*len(target_pollutants)), sharex=True)
for i, pol in enumerate(target_pollutants):
    axes[i].plot(dates_future, pred_next_24h[:, i], label=f"Prédit {pol}", marker='x', linestyle='--', color='orange')
    axes[i].set_ylabel(pol)
    axes[i].legend()
    axes[i].grid(True)
axes[-1].set_xlabel("Date / Heure")
plt.suptitle("Prédiction des 24 prochaines heures par polluant", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("\n--- Prédictions pour les 24 prochaines heures ---")
for i, dt in enumerate(dates_future):
    values = pred_next_24h[i]
    print(f"{dt}: ", end="")
    for pol, val in zip(target_pollutants, values):
        print(f"{pol}={val:.2f}", end="  ")
    print()
