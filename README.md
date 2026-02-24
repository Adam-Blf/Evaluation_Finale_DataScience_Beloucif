# 📊 Projet Évaluation Finale - Data Science avec Python

```text
==========================================================================
AUTEUR      : Adam Beloucif
FORMATION   : Mastère Data Science
PROJET      : Prédiction du Churn Telco
DATE        : Février 2026
==========================================================================
```

## 🚀 Présentation

Ce projet présente une pipeline complète de Data Science pour la prédiction de la perte de clients (Churn) dans le secteur des télécommunications.

## 📁 Structure du Projet

- `data/` : Jeu de données source (`telco_churn.csv`).
- `api/` : Code source de l'API FastAPI et modèles sérialisés (`.pkl`).
- `scripts/` : Scripts d'entraînement et utilitaires.
- `reports/` : Graphiques d'EDA et Rapport Final PDF.
- `Dockerfile` : Pour la conteneurisation.

## 🛠️ Installation & Exécution

1. Installez les dépendances : `pip install -r requirements.txt`
2. Lancez l'API : `python api/main.py`
3. Testez l'endpoint : `POST /predict` avec les données client.

## 📊 Caractéristiques Techniques

- **Modèle Supervisé** : Random Forest (Classification).
- **Modèle Non Supervisé** : K-Means (Segmentation).
- **API** : FastAPI avec validation Pydantic.
- **Reporting** : Rapport PDF généré automatiquement.

---
*Livrable réalisé par Adam Beloucif.*
