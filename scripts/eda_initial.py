"""
==========================================================================
TITRE       : Analyse Exploratoire des Données (EDA) - Projet Churn Telco
AUTEUR      : Adam Beloucif
PROJET      : Évaluation Finale Data Science Python
DATE        : 24 Février 2026
DESCRIPTION : Exploration, nettoyage et analyse statistique du dataset Telco.
==========================================================================
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration de l'encodage pour la console Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Configuration du logging professionnel
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [INFO] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def perform_eda(file_path):
    logger.info("🚀 Lancement de l'analyse exploratoire des données...")
    
    if not os.path.exists(file_path):
        logger.error(f"❌ Fichier non trouvé : {file_path}")
        return

    # Chargement des données
    df = pd.read_csv(file_path)
    logger.info(f"📊 Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes.")

    # 1. Structure du dataset
    logger.info("🔍 Analyse de la structure des données...")
    print("\n--- Aperçu des 5 premières lignes ---")
    print(df.head())
    
    print("\n--- Types de colonnes ---")
    print(df.dtypes)

    # 2. Valeurs manquantes et Outliers
    logger.info("🧐 Recherche de valeurs manquantes et incohérentes...")
    # La colonne TotalCharges est souvent objet alors qu'elle devrait être numérique
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing_values = df.isnull().sum()
    logger.info(f"⚠️ Valeurs manquantes identifiées :\n{missing_values[missing_values > 0]}")
    
    # Remplissage des TotalCharges manquantes par 0 (clients ayant 0 mois d'ancienneté)
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # 3. Statistiques descriptives
    logger.info("📈 Calcul des statistiques descriptives...")
    print("\n--- Statistiques (Numérique) ---")
    print(df.describe())

    # 4. Visualisations
    logger.info("🎨 Génération des graphiques d'analyse...")
    sns.set_theme(style="whitegrid")
    
    # Distribution du Churn
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Churn', data=df, palette='viridis')
    plt.title('Distribution de la Résiliation (Churn)')
    plt.savefig('reports/churn_distribution.png')
    logger.info("✅ Graphique de distribution du Churn sauvegardé.")

    # Corrélation des variables numériques
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matrice de Corrélation')
    plt.savefig('reports/correlation_matrix.png')
    logger.info("✅ Matrice de corrélation sauvegardée.")

    logger.info("🏁 Fin de l'analyse exploratoire.")
    return df

if __name__ == "__main__":
    DATA_PATH = "data/telco_churn.csv"
    df_cleaned = perform_eda(DATA_PATH)
