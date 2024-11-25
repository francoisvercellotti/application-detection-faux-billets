# Utilise une image Python légère
FROM python:3.10-slim

# Définit le répertoire de travail à l'intérieur du conteneur
WORKDIR /app

# Copie uniquement le fichier requirements.txt pour éviter de re-télécharger les dépendances à chaque changement de code
COPY requirements.txt /app/

# Met à jour pip et installe les dépendances
RUN pip install --upgrade pip && pip install --no-cache-dir -v -r requirements.txt



# Copie le reste de l'application et les modèles
COPY . /app/

# Exposer le port 80 pour accéder à l'application
EXPOSE 80

# Commande pour démarrer l'application Streamlit
CMD ["streamlit", "run", "/app/appli.py", "--server.port=80", "--server.address=0.0.0.0"]
