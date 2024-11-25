# Utilise une image Python légère
FROM python:3.9-slim

# Définit le répertoire de travail à l'intérieur du conteneur
WORKDIR /app

# Copie le script de prédiction, les dépendances et le modèle dans le conteneur
COPY scripts/predict_from_file.py /app/scripts/
COPY model/best_model.pkl /app/model/
COPY requirements.txt /app/

# Installe les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Définit le point d'entrée par défaut pour le conteneur
ENTRYPOINT ["python", "/app/scripts/predict_from_file.py"]
