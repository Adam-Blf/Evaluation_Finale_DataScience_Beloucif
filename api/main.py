"""
==========================================================================
TITRE       : API Avancée de Prédiction du Churn - Telco
AUTEUR      : Adam Beloucif
PROJET      : Évaluation Finale Data Science Python
DATE        : 24 Février 2026
DESCRIPTION : Serveur FastAPI utilisant le modèle Random Forest avancé.
==========================================================================
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import joblib
import pandas as pd
import os
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Telco Churn Advanced API", version="2.0.0")

# Chargement des modèles avancés
MODEL_PATH = "model_advanced.pkl"
SCALER_PATH = "scaler_advanced.pkl"
FEATURES_PATH = "features_advanced.pkl"

if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "api/model_advanced.pkl"
    SCALER_PATH = "api/scaler_advanced.pkl"
    FEATURES_PATH = "api/features_advanced.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features_list = joblib.load(FEATURES_PATH)
    logger.info("✅ Modèles avancés chargés avec succès.")
except Exception as e:
    logger.error(f"❌ Erreur lors du chargement des modèles : {str(e)}")

class CustomerDataAdvanced(BaseModel):
    # Modèle complet basé sur les features attendues après preprocessing
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float
    Tenure_Group: int  # Nouvelle feature encodée
    
    model_config = ConfigDict(protected_namespaces=())

@app.get("/")
def read_root():
    return {"status": "online", "model": "Random Forest Advanced (Balanced)", "version": "2.0.0"}

@app.post("/predict")
def predict_churn(data: CustomerDataAdvanced):
    try:
        input_data = pd.DataFrame([data.model_dump()])
        
        # Vérification des colonnes
        missing_cols = set(features_list) - set(input_data.columns)
        if missing_cols:
            raise ValueError(f"Colonnes manquantes : {missing_cols}")
            
        # Réordonner selon le modèle
        input_data = input_data[features_list]
        
        # Scaling
        input_scaled = scaler.transform(input_data)
        
        # Prédiction
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]
        
        return {
            "churn_prediction": int(prediction[0]),
            "churn_probability": float(probability),
            "status": "success",
            "model_used": "Advanced Balanced RF"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
