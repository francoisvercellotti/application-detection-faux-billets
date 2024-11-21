"""
Module pour entraîner et évaluer un modèle de régression logistique
avec recherche d'hyperparamètres, prétraitement et sauvegarde des résultats.

Ce module contient une fonction principale `train_test_split_and_tune` qui :
1. Sépare les données en ensembles d'entraînement et de test.
2. Effectue un prétraitement comprenant une transformation polynomiale
   et une sélection de variables.
3. Entraîne un modèle de régression logistique avec une recherche
d'hyperparamètres via GridSearchCV.
4. Sauvegarde le meilleur modèle, les transformateurs utilisés,
   et les jeux de données d'entraînement et de test.

Les résultats du modèle sont sauvegardés dans un fichier pickle et
les transformateurs dans des fichiers séparés.
Les jeux de données sont également sauvegardés dans un format compressé `.npz`.

Fonctionnalités :
- Séparation des données en ensemble d'entraînement et de test.
- Recherche d'hyperparamètres pour la régression logistique.
- Prétraitement des données incluant la transformation polynomiale et
  la sélection de variables.
- Sauvegarde des objets du modèle et des transformateurs pour un déploiement ultérieur.

Exécution :
- Lors de l'exécution du module, un modèle sera entraîné
  à partir du DataFrame nettoyé et prétraité fourni.
- Les résultats seront enregistrés dans des fichiers spécifiés par l'utilisateur.
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

def train_test_split_and_tune(
    input_df,  # DataFrame d'entrée contenant les données à traiter
    output_model_path="model/best_model.pkl",  # Chemin pour enregistrer le meilleur modèle
    output_transformers_dir="model",  # Répertoire pour sauvegarder les transformateurs
    hyper_params=None,  # Dictionnaire optionnel des hyperparamètres pour la recherche
):
    """
    Sépare les données en ensembles d'entraînement et de test, entraîne un modèle
    de régression logistique avec recherche d'hyperparamètres et sauvegarde les
    résultats du modèle et des transformations.

    Arguments:
        input_df (DataFrame): DataFrame contenant les données.
        output_model_path (str): Chemin pour enregistrer le meilleur modèle.
        output_transformers_dir (str): Répertoire pour sauvegarder les transformateurs.
        hyper_params (dict, optionnel): Dictionnaire des hyperparamètres pour la recherche.

    Retourne:
        best_estimator (estimator): Le meilleur estimateur trouvé par la recherche.
        best_params (dict): Les meilleurs paramètres trouvés.
        x_train_data (ndarray): Variables d'entraînement.
        y_train_data (ndarray): Cible d'entraînement.
        x_test_data (ndarray): Variables de test.
        y_test_data (ndarray): Cible de test.
        grid_search (GridSearchCV): L'objet GridSearchCV ajusté.
    """

    # Créer le dossier de sauvegarde si nécessaire
    os.makedirs(output_transformers_dir, exist_ok=True)

    # Séparer les données en features (X) et cible (y)
    x_data = input_df.drop(columns=['is_genuine'])  # 'is_genuine' est la colonne cible
    y_data = input_df['is_genuine']

    # Diviser les données en ensembles d'entraînement et de test
    x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(
        x_data, y_data, test_size=0.2, random_state=0)

    # Définir le pipeline de prétraitement
    preprocessor = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('select', SelectKBest(f_classif, k=10)),
    ])

    # Définir le pipeline de classification
    logistic_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('logistic', LogisticRegression(random_state=0, max_iter=20000))
    ])

    # Définir la grille d'hyperparamètres par défaut
    if hyper_params is None:
        hyper_params = {
            'preprocessor__poly__degree': [2, 3],
            'preprocessor__select__k': range(6, min(84, x_train_split.shape[1] + 1)),
            'logistic__penalty': ['l1', 'l2', 'None'],
            'logistic__C': [0.01, 0.1, 1, 10, 100],
            'logistic__solver': ['saga', 'liblinear', 'lbfgs'],
        }

    # Configurer GridSearchCV
    grid_search_cv = GridSearchCV(
        logistic_pipeline, hyper_params, scoring='f1', cv=4, n_jobs=-1, verbose=1)

    grid_search_cv.fit(x_train_split, y_train_split)

    # Obtenir le meilleur estimateur et ses paramètres
    best_estimator = grid_search_cv.best_estimator_
    best_params = grid_search_cv.best_params_

    print("Meilleurs hyperparamètres trouvés :")
    print(best_params)

    # Sauvegarder le meilleur modèle
    joblib.dump(best_estimator, output_model_path)
    print(f"Meilleur modèle sauvegardé dans : {output_model_path}")

    # Sauvegarder les transformateurs individuels pour le déploiement
    joblib.dump(best_estimator.named_steps['preprocessor'].named_steps['poly'],
                f"{output_transformers_dir}/transformer/poly.pkl")
    joblib.dump(best_estimator.named_steps['preprocessor'].named_steps['select'],
                f"{output_transformers_dir}/transformer/selector.pkl")
    joblib.dump(best_estimator.named_steps['scaler'],
                f"{output_transformers_dir}/transformer/scaler.pkl")

    # Sauvegarder les datasets pour une évaluation ultérieure
    np.savez_compressed(
    "data/derived/train_test_data.npz",  # Corriger les guillemets
    x_train_data=x_train_split,
    y_train_data=y_train_split,
    x_test_data=x_test_split,
    y_test_data=y_test_split
    )
    print(
        "Jeux d'entraînement et de test sauvegardés dans : "
        "data/derived/train_test_data.npz"
    )


    return (best_estimator, best_params, x_train_split, y_train_split,
            x_test_split, y_test_split, grid_search_cv)

if __name__ == "__main__":
    # Charger le DataFrame nettoyé
    df_input = pd.read_csv(
        'data/derived/cleaned_and_encoded_dataset.csv', delimiter=';', skipinitialspace=True)

    # Appeler la fonction pour entraîner le modèle et effectuer la recherche d'hyperparamètres
    best_logistic_estimator, best_logistic_params, x_train_data,\
    y_train_data, x_test_data, y_test_data,\
    logistic_grid_search_cv = train_test_split_and_tune(df_input)
