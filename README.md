# 📊 Projet Évaluation Finale - Data Science avec Python

[![version](https://img.shields.io/badge/version-0.1.0-000091?style=flat-square)](https://github.com/Adam-Blf/Evaluation_Finale_DataScience_Beloucif/releases)

<!-- adam-badges:start -->
[![commits](https://img.shields.io/github/commit-activity/t/Adam-Blf/Evaluation_Finale_DataScience_Beloucif?color=001329&label=commits&style=flat-square)](https://github.com/Adam-Blf/Evaluation_Finale_DataScience_Beloucif/commits) [![visites](https://hits.sh/github.com/Adam-Blf/Evaluation_Finale_DataScience_Beloucif.svg?style=flat-square&label=visites&color=001329)](https://hits.sh/github.com/Adam-Blf/Evaluation_Finale_DataScience_Beloucif/) [![last commit](https://img.shields.io/github/last-commit/Adam-Blf/Evaluation_Finale_DataScience_Beloucif?color=D4A437&style=flat-square&label=dernier%20push)](https://github.com/Adam-Blf/Evaluation_Finale_DataScience_Beloucif/commits) [![top language](https://img.shields.io/github/languages/top/Adam-Blf/Evaluation_Finale_DataScience_Beloucif?style=flat-square)](https://github.com/Adam-Blf/Evaluation_Finale_DataScience_Beloucif) [![license](https://img.shields.io/github/license/Adam-Blf/Evaluation_Finale_DataScience_Beloucif?style=flat-square&color=D4A437)](LICENSE)
<!-- adam-badges:end -->


[![EFREI Paris](https://img.shields.io/badge/EFREI-Paris-005CA9?style=flat-square&labelColor=000000)](https://www.efrei.fr/)

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

## Architecture

```mermaid
flowchart TB
    A["data/telco_churn.csv<br/>jeu de donnees source"]
    B["eda.ipynb - scripts/eda_initial.py<br/>analyse exploratoire"]
    C["scripts/train_supervised.py<br/>RandomForest - classification churn"]
    D["scripts/train_unsupervised.py<br/>K-Means - segmentation clients"]
    E["api/*.pkl<br/>model - scaler - features serialises"]
    F["api/main.py<br/>FastAPI - validation Pydantic"]
    G["POST /predict<br/>prediction churn client"]
    H["tests/test_api.py<br/>tests endpoint"]
    I["reports/<br/>graphiques EDA - rapport PDF"]
    J["Dockerfile<br/>conteneurisation"]
    A --> B
    B --> C
    B --> D
    C --> E
    D --> E
    E --> F
    F --> G
    F --> H
    B --> I
    F --> J
```

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
  <sub>Par <a href="https://adam.beloucif.com">Adam Beloucif</a> - Data Engineer & Fullstack Developer - <a href="https://github.com/Adam-Blf">GitHub</a> - <a href="https://www.linkedin.com/in/adambeloucif/">LinkedIn</a></sub>
</p>


## Star History

<a href="https://www.star-history.com/?repos=Adam-Blf%2FEvaluation_Finale_DataScience_Beloucif&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=Adam-Blf/Evaluation_Finale_DataScience_Beloucif&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=Adam-Blf/Evaluation_Finale_DataScience_Beloucif&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=Adam-Blf/Evaluation_Finale_DataScience_Beloucif&type=date&legend=top-left" />
 </picture>
</a>
