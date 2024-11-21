# Règle de nettoyage
clean:
	rm -rf data/* model/*

# Définir le shell
SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

# Créer les dossiers nécessaires
data:
	mkdir -p data/Input data/Intermediate data/Processed model

# Étape 1 : Charger les données (en utilisant l'argument FILE)
data/loaded_dataset.csv: data/Input/raw_data
	@ if [ -z "$(FILE)" ]; then echo "Erreur : il faut spécifier le chemin du fichier à charger avec FILE=<chemin>"; exit 1; fi
	python3 scripts/load_data.py $(FILE)  # Charger le fichier spécifié par FILE (ex: billets.csv)

# Étape 2 : Nettoyer les données
data/cleaned_and_encoded_dataset.csv: data/loaded_dataset.csv
	python3 scripts/clean_and_encode_data.py

# Étape 3 & 4 : Prétraitement + Entraînement du modèle
model/best_model.pkl: data/cleaned_and_encoded_dataset.csv
	python3 scripts/train_test_split_and_tune.py

# Étape 5 : Évaluer le modèle sur les jeux d'entraînement et de test
evaluate:
	python3 scripts/model_evaluation_metrics_and_visualizations.py model/best_model.pkl data/cleaned_and_encoded_dataset.csv

# Étape 6 : Prédiction à partir des fichiers fournis par l'utilisateur
predict:
	@ if [ -z "$(INPUT_FILE)" ]; then echo "Erreur : il faut spécifier le fichier à prédire avec INPUT_FILE=<chemin>"; exit 1; fi
	@ if [ -z "$(MODEL_PATH)" ]; then echo "Erreur : il faut spécifier le chemin du modèle avec MODEL_PATH=<chemin>"; exit 1; fi
	@ if [ -z "$(OUTPUT_FILE)" ]; then echo "Erreur : il faut spécifier le fichier de sortie avec OUTPUT_FILE=<chemin>"; exit 1; fi
	python3 scripts/predict_from_file.py $(INPUT_FILE) $(MODEL_PATH) $(OUTPUT_FILE)

# Commande principale pour exécuter toutes les étapes
all: clean data/loaded_dataset.csv data/cleaned_and_encoded_dataset.csv model/best_model.pkl evaluate_train evaluate_test
