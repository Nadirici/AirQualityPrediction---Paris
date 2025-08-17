````markdown
# Prédiction de la pollution horaire à Paris

Ce projet permet de prédire les 24 prochaines heures de pollution sur Paris en utilisant un modèle LSTM entraîné sur des données horaires de pollution et météo. Le modèle est plus précis pour l'heure suivante, car il a été conçu sur des séquences horaires.

## Structure du projet

- `db/pollution_db.sql` : export de la base de données MySQL contenant les mesures de pollution et météo.  
- `model/` : dossier contenant les modèles et scalers sauvegardés.  
- `pipeline_data.py` : script de préparation et traitement des données pour l'entraînement.  
- `model.ipynb` : notebook pour l'entraînement du modèle.  
- `predict.py` : script pour générer les prédictions des 24 prochaines heures.  

## Setup

### 1. Installation des dépendances

```bash
pip install -r requirements.txt
````

### 2. Configuration de la base de données

1. Installer WAMP (ou tout serveur MySQL).
2. Ouvrir **phpMyAdmin** et créer une base `pollution_db`.
3. Importer le fichier SQL `db/pollution_db.sql` pour recréer la base avec les données.
4. Créer un fichier `.env` à la racine du projet pour sécuriser les identifiants :

```
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=votre_mot_de_passe
DB_NAME=pollution_db
```

> Ne pas pousser le fichier `.env` sur GitHub.

### 3. Préparation des données

```bash
python pipeline_data.py
```

Ce script nettoie, agrège et prépare les séquences horaires pour l'entraînement.

### 4. Entraînement du modèle

Ouvrir le notebook `model.ipynb` et exécuter les cellules pour entraîner le modèle sur les séquences préparées. Les scalers et le modèle entraîné seront sauvegardés dans `model/`.

### 5. Prédictions des 24 prochaines heures

```bash
python predict.py
```

Le script utilise le modèle entraîné pour prédire les 24 prochaines heures de pollution. Comme le modèle est basé sur des séquences horaires, il est particulièrement précis pour la **prochaine heure**.

## Remarques

* Les fichiers `.env` et la base de données ne doivent pas être versionnés pour des raisons de sécurité.
* Le modèle peut être réentraîné régulièrement avec de nouvelles données pour améliorer ses performances.


