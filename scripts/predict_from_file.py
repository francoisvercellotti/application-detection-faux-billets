"""
Script pour effectuer des prédictions sur des billets de banque.

Ce script charge des données depuis un fichier, applique un modèle de machine learning
pré-entraîné pour prédire si les billets sont vrais ou faux, et sauvegarde les résultats
dans un fichier de sortie.

Exemple d'utilisation :
    python3 scripts/predict_from_file.py <input_file> <model_path> <output_file>
"""

import os
import argparse
import joblib
import pandas as pd


def load_data(file_path, delimiter=',', **kwargs):
    """
    Charge les données d'un fichier en fonction de son extension.

    Args:
        file_path (str): Chemin vers le fichier à charger.
        delimiter (str, optional): Délimiteur utilisé pour les fichiers CSV ou texte.
        Par défaut ','.
        **kwargs: Arguments supplémentaires pour les fonctions de chargement.

    Returns:
        pd.DataFrame: Un DataFrame contenant les données chargées, limité aux colonnes nécessaires.

    Raises:
        ValueError: Si l'extension du fichier n'est pas supportée.
    """

    # Obtenir l'extension du fichier
    _, file_extension = os.path.splitext(file_path)

    # Charger le fichier en fonction de son extension
    if file_extension == '.csv':
        data = pd.read_csv(file_path, delimiter=delimiter, **kwargs)
    elif file_extension in ['.xls', '.xlsx']:
        data = pd.read_excel(file_path, **kwargs)
    elif file_extension == '.json':
        data = pd.read_json(file_path, **kwargs)
    elif file_extension == '.parquet':
        data = pd.read_parquet(file_path, **kwargs)
    elif file_extension == '.txt':
        # Si le fichier texte utilise le délimiteur fourni
        data = pd.read_csv(file_path, delimiter=delimiter, **kwargs)
    else:
        raise ValueError(f"Extension de fichier '{file_extension}' non supportée.")

    # Garder uniquement les colonnes nécessaires
    columns_needed = ['id','diagonal', 'height_left', 'height_right',\
        'margin_low', 'margin_up', 'length']
    data = data[[col for col in columns_needed if col in data.columns]]
    return data

def load_model(model_path):
    # Charger le modèle à partir du fichier .pkl
    model = joblib.load(model_path)
    return model

def predict_new_data(model, new_data):
    """
    Effectue des prédictions sur de nouvelles données à l'aide d'un modèle entraîné.

    Args:
        model: Modèle chargé pour effectuer les prédictions.
        new_data (pd.DataFrame): Les nouvelles données, incluant une colonne 'id'.

    Returns:
        pd.DataFrame: Un DataFrame contenant les colonnes 'id' et 'Prediction'.
    """

    # Extraire la colonne 'id' et préparer les données pour la prédiction
    data_for_prediction = new_data.drop(columns=['id'])

    # Faire des prédictions avec le modèle
    predictions = model.predict(data_for_prediction)

    # Ajouter les prédictions au DataFrame
    new_data['Prediction'] = predictions
    new_data['Prediction'] = new_data['Prediction'].map({1: 'Vrai', 0: 'Faux'})

    # Garder uniquement les colonnes 'id' et 'Prediction' dans le DataFrame de sortie
    return new_data[['id', 'Prediction']]

def main_predict():
    """
    Point d'entrée principal pour exécuter le script.

    Charge les données, effectue les prédictions et sauvegarde les résultats.
    """

   # Créer un parseur d'arguments
    parser = argparse.ArgumentParser(description="Prédiction des billets.")

    # Ajouter des arguments pour les chemins d'entrée et de sortie
    parser.add_argument('input_file', type=str, help="Chemin vers le fichier d'entrée.")
    parser.add_argument('model_path', type=str, help="Chemin vers le fichier du modèle.")
    parser.add_argument('output_file', type=str, help="Chemin vers le fichier de sortie.")

    # Analyser les arguments
    args = parser.parse_args()

    # Charger les données
    df = load_data(args.input_file)

    # Charger le modèle
    model = load_model(args.model_path)

    # Faire des prédictions sur les nouvelles données
    predictions_df = predict_new_data(model, df)

    # Sauvegarder les résultats dans un fichier CSV
    predictions_df.to_csv(args.output_file, index=False, sep=';')
    print(f"Les prédictions ont été sauvegardées dans : {args.output_file}")

if __name__ == "__main__":
    main_predict()
