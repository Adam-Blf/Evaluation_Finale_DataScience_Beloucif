# 📊 Projet Évaluation Finale - Data Science avec Python

![Status](https://img.shields.io/badge/status-academic-blue)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)

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


---

<p align="center">
  <sub>Par <a href="https://adam.beloucif.com">Adam Beloucif</a> · Data Engineer & Fullstack Developer · <a href="https://github.com/Adam-Blf">GitHub</a> · <a href="https://www.linkedin.com/in/adambeloucif/">LinkedIn</a></sub>
</p>
