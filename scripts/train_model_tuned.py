"""
==========================================================================
TITRE       : Hyperparameter Tuning (GridSearchCV) - Telco Churn
AUTEUR      : Adam Beloucif
PROJET      : Évaluation Finale Data Science Python
DATE        : 24 Février 2026
DESCRIPTION : Recherche exhaustive des meilleurs paramètres pour le Random Forest 
              afin de contrer l'overfitting et maximiser la détection.
==========================================================================
"""

import sys
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Configuration console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Logging professionnel
logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def evaluate_and_tune(file_path):
    logger.info("📦 Chargement des données et préparation...")
    df = pd.read_csv(file_path)
    df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # Feature Engineering
    df['Tenure_Group'] = df['tenure'].apply(
        lambda x: '0-1_an' if x <= 12 else ('1-2_ans' if x <= 24 else ('2-4_ans' if x <= 48 else 'plus_4_ans'))
    )
    
    # Encodage
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
        
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    logger.info("🔪 Split (80/20) et Standardisation...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- HYPERPARAMETER TUNING ---
    logger.info("⚙️ Configuration de la grille de recherche (GridSearchCV)...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    # Recherche avec Validation Croisée (CV=3 pour la rapidité ici, 5 idéalement)
    logger.info("🚀 Lancement du GridSearch (cela peut prendre du temps)...")
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    
    logger.info(f"🏆 Meilleurs hyperparamètres trouvés : {grid_search.best_params_}")
    
    # Modèle optimal
    best_model = grid_search.best_estimator_
    
    # Évaluation
    y_pred = best_model.predict(X_test_scaled)
    logger.info("📊 Évaluation Finale du modèle optimisé :")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    # Remplacement de l'ancien modèle
    logger.info("💾 Sauvegarde du modèle 'Tuned' en tant que 'model_advanced.pkl'...")
    joblib.dump(best_model, 'api/model_advanced.pkl')
    # Scaler et Features ne changent pas par rapport au pipeline avancé précédent
    
    logger.info("✅ Excellence Technique : Tuning complété avec succès.")

if __name__ == "__main__":
    DATA_PATH = "data/telco_churn.csv"
    evaluate_and_tune(DATA_PATH)
