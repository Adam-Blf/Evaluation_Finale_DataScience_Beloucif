# Utilisation d'une image Python légère
FROM python:3.9-slim

# Répertoire de travail
WORKDIR /app

# Installation des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie des fichiers de l'API et des modèles
COPY api/ /app/

# Exposer le port de FastAPI
EXPOSE 8000

# Commande de lancement
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
