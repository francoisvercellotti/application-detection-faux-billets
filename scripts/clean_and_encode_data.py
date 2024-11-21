"""
Script pour nettoyer et encoder un jeu de données.

Ce script charge un jeu de données, encode les variables booléennes en entiers,
supprime les lignes contenant des valeurs manquantes, et enregistre le jeu de
données nettoyé dans un nouveau fichier CSV.
"""

import pandas as pd

def clean_and_encode_data():
    """
    Nettoie et encode le jeu de données :
    - Charge les données depuis 'data/loaded_dataset.csv'
    - Encode les variables booléennes en entiers (False -> 0, True -> 1)
    - Supprime les lignes contenant des valeurs manquantes
    - Sauvegarde le jeu de données nettoyé dans 'data/cleaned_and_encoded_dataset.csv'
    """
    # Charger les données
    data = pd.read_csv('data/loaded/loaded_dataset.csv', delimiter=";")

    # Encoder la variable cible
    code = {False: 0, True: 1}
    for col in data.select_dtypes("bool"):
        data[col] = data[col].map(code)

    # Afficher un résumé des données avant nettoyage
    print("Avant nettoyage :")
    print(data.isnull().sum())  # Afficher le nombre de valeurs manquantes dans chaque colonne

    # Exécuter les étapes de nettoyage (suppression des valeurs manquantes)
    data = data.dropna(how="any")

    # Afficher un résumé des données après nettoyage
    print("\nAprès nettoyage :")
    print(data.isnull().sum())  # Vérifier que toutes les valeurs manquantes ont été supprimées

    # Sauvegarder le fichier nettoyé
    data.to_csv('data/derived/cleaned_and_encoded_dataset.csv', index=False, sep=";")
    print("\nLe fichier nettoyé a été sauvegardé sous 'data/cleaned_and_encoded_dataset.csv'.")


if __name__ == "__main__":
    clean_and_encode_data()
