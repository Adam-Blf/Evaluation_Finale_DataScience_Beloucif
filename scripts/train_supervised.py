"""
==========================================================================
TITRE       : Modélisation Supervisée - Projet Churn Telco
AUTEUR      : Adam Beloucif
PROJET      : Évaluation Finale Data Science Python
DATE        : 24 Février 2026
DESCRIPTION : Nettoyage des données, encodage et entraînement Random Forest.
==========================================================================
"""

import sys
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Configuration console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')
logger = logging.getLogger(__name__)

def train_supervised_model(file_path):
    logger.info("📦 Chargement et nettoyage final des données...")
    df = pd.read_csv(file_path)
    
    # Nettoyage TotalCharges (caractères vides -> numérique)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Suppression customerID (inutile pour le modèle)
    df = df.drop('customerID', axis=1)
    
    # Encodage des variables catégorielles
    logger.info("🔐 Encodage des variables catégorielles...")
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    
    # Séparation Features / Target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Mise à l'échelle (Standardisation) indispensable pour certains modèles, optionnel pour RF mais recommandé
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modélisation Supervisée : Random Forest
    logger.info("🚀 Entraînement du modèle Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Évaluation
    y_pred = rf_model.predict(X_test_scaled)
    logger.info("📊 Évaluation du modèle :")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    # Sauvegarde des objets
    joblib.dump(rf_model, 'api/model.pkl')
    joblib.dump(scaler, 'api/scaler.pkl')
    joblib.dump(X.columns.tolist(), 'api/features.pkl')
    
    logger.info("✅ Modèle, Scaler et Features sauvegardés dans le dossier 'api/'.")
    return rf_model

if __name__ == "__main__":
    DATA_PATH = "data/telco_churn.csv"
    train_supervised_model(DATA_PATH)
