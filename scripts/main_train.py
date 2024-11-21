"""
Module principal pour le chargement, le nettoyage, le fractionnement des données,
l'entraînement du modèle et l'évaluation des performances du modèle.

Ce module gère le processus complet de préparation des données et d'évaluation du modèle :
1. Chargement des données brutes depuis un fichier.
2. Nettoyage et encodage des données.
3. Fractionnement des données en ensembles d'entraînement et de test.
4. Entraînement du modèle et réglage des hyperparamètres.
5. Évaluation du modèle et génération de visualisations.

"""

import sys
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from load_data import load_data
from clean_and_encode_data import clean_and_encode_data
from train_test_split_and_tune import train_test_split_and_tune
from model_evaluation_metrics_and_visualizations\
    import model_evaluation_metrics_and_visualizations


def main():
    """
    Fonction principale pour charger, nettoyer, diviser les données, entraîner le modèle
    et évaluer ses performances.
    """
    # Étape 1 : Charger les données brutes
    if len(sys.argv) < 2:
        print("Erreur : veuillez fournir le chemin du fichier de données à charger.")
        sys.exit(1)

    # Récupérer le chemin du fichier passé en argument
    raw_file_path = sys.argv[1]

    # Charger les données
    print("Chargement des données...")
    load_data(raw_file_path, delimiter=';')

    # Étape 2 : Nettoyage et encodage des données
    print("Nettoyage et encodage des données...")
    clean_and_encode_data()

    # Étape 3 : Diviser les données et ajuster le modèle
    print("Division des données et tuning du modèle...")
    df_input=pd.read_csv(
        'data/derived/cleaned_and_encoded_dataset.csv', delimiter=';', skipinitialspace=True)
    train_test_split_and_tune(df_input)

    # Étape 4 : Évaluation et visualisation
    print("Évaluation et génération des visualisations...")
    # Charger le modèle déjà entraîné et sauvegardé
    best_model = joblib.load("model/best_model.pkl")

    # Charger les données de test et d'entraînement
    npz_file = np.load("data/derived/train_test_data.npz")
    df_input = {
        'x_train': npz_file['x_train_data'],
        'y_train': npz_file['y_train_data'],
        'x_test': npz_file['x_test_data'],
        'y_test': npz_file['y_test_data']
    }

    # Générer le timestamp actuel
    actual_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = "logistic_model"  # Nom du fichier

    # Appeler la fonction pour évaluer le modèle et générer les visualisations
    model_evaluation_metrics_and_visualizations(
        best_model,
        df_input,
        actual_timestamp,
        filename
    )

    print("Évaluation terminée. Visualisations et rapports générés.")

if __name__ == "__main__":
    main()
