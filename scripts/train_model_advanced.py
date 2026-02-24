"""
==========================================================================
TITRE       : Modèle de Prédiction Avancé (Churn Telco)
AUTEUR      : Adam Beloucif
PROJET      : Évaluation Finale Data Science Python
DATE        : 24 Février 2026
DESCRIPTION : Pipeline ML complet incluant Feature Engineering, gestion de
              l'imbalanced data et Random Forest Classifier.
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

# Logging professionnel
logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def advanced_pipeline(file_path):
    logger.info("📦 Démarrage du pipeline de Machine Learning Avancé...")
    
    # 1. Chargement & Nettoyage de base
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"❌ Impossible de trouver {file_path}")
        return
        
    df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # 2. Feature Engineering
    logger.info("⚙️ Application du Feature Engineering...")
    # Regroupement de l'ancienneté
    def group_tenure(month):
        if month <= 12: return '0-1_an'
        elif month <= 24: return '1-2_ans'
        elif month <= 48: return '2-4_ans'
        else: return 'plus_4_ans'
    
    df['Tenure_Group'] = df['tenure'].apply(group_tenure)
    
    # 3. Préparation (Encodage)
    logger.info("🔐 Encodage des variables pour les algorithmes mathématiques...")
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=['object']).columns
    
    label_encoders = {}
    for col in cat_cols:
        l_enc = LabelEncoder()
        df[col] = l_enc.fit_transform(df[col])
        label_encoders[col] = l_enc
    
    # 4. Séparation X/y
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # 5. Split Train/Test
    logger.info("🔪 Séparation temporelle train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 6. Scaling (Standardisation)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 7. Modélisation avec gestion de l'imbálance
    logger.info("🤖 Entraînement du Random Forest (avec gestion du déséquilibre)...")
    # POURQUOI class_weight='balanced' ? Le ratio Churn/No-Churn est de 26/74. 
    # Sans cette option, le modèle aurait tendance à toujours prédire "No-Churn" pour maximiser sa précision globale,
    # au détriment de la détection des vrais Churners.
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train_scaled, y_train)
    
    # 8. Évaluation
    y_pred = rf_model.predict(X_test_scaled)
    logger.info("📊 Évaluation Finale :")
    print("\n--- Rapport de Classification ---")
    print(classification_report(y_test, y_pred, target_names=['No Churn (0)', 'Churn (1)']))
    
    # 9. Sauvegarde
    joblib.dump(rf_model, 'api/model_advanced.pkl')
    joblib.dump(scaler, 'api/scaler_advanced.pkl')
    # On sauvegarde les colonnes exactes pour l'API
    joblib.dump(X.columns.tolist(), 'api/features_advanced.pkl')
    
    logger.info("✅ Modèle Avancé, Transformateurs et Features sauvegardés dans 'api/'.")

if __name__ == "__main__":
    DATA_PATH = "data/telco_churn.csv"
    advanced_pipeline(DATA_PATH)
