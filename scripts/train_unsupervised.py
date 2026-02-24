"""
==========================================================================
TITRE       : Modélisation Non Supervisée - Projet Churn Telco
AUTEUR      : Adam Beloucif
PROJET      : Évaluation Finale Data Science Python
DATE        : 24 Février 2026
DESCRIPTION : Segmentation des clients via K-Means (Clustering).
==========================================================================
"""

import sys
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Configuration console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')
logger = logging.getLogger(__name__)

def train_unsupervised_model(file_path):
    logger.info("📦 Préparation des données pour le Clustering...")
    df = pd.read_csv(file_path)
    
    # Sélection des variables numériques pour le clustering (comportement d'achat)
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    X = df[num_cols]
    
    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means
    logger.info("🚀 Lancement de l'algorithme K-Means (3 clusters)...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Évaluation
    score = silhouette_score(X_scaled, clusters)
    logger.info(f"📊 Score de Silhouette : {score:.4f}")
    
    # Sauvegarde
    joblib.dump(kmeans, 'api/kmeans_model.pkl')
    logger.info("✅ Modèle K-Means sauvegardé.")
    
    return kmeans

if __name__ == "__main__":
    DATA_PATH = "data/telco_churn.csv"
    train_unsupervised_model(DATA_PATH)
