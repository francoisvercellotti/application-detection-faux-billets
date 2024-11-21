"""
Ce module contient des fonctions pour évaluer les performances des modèles,
afficher des métriques et des visualisations.
"""

import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, precision_recall_curve,confusion_matrix
import joblib
import numpy as np


def save_classification_reports(train_report, test_report, evaluation_dir, filename):
    """
    Fonction pour sauvegarder les rapports de classification dans un fichier texte.
    """
    with open(f"{evaluation_dir}/{filename}_metrics.txt", "w", encoding="utf-8") as f:
        f.write("### Métriques d'évaluation ###\n")
        f.write("Rapport de classification sur le jeu d'entraînement:\n")
        f.write(train_report + "\n\n")
        f.write("Rapport de classification sur le jeu de test:\n")
        f.write(test_report)


def plot_confusion_matrix(y_true, y_pred, evaluation_dir, filename):
    """
    Fonction pour générer et sauvegarder la matrice de confusion.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Contrefait', 'Authentique'], yticklabels=['Contrefait', 'Authentique'])
    plt.title('Matrice de Confusion')
    plt.xlabel('Prédictions')
    plt.ylabel('Vérités')
    plt.savefig(f"{evaluation_dir}/{filename}_confusion_matrix.png")
    plt.close()

def plot_precision_recall_curve(precision, recall, thresholds, evaluation_dir, filename):
    """
    Fonction pour générer et sauvegarder la courbe de précision-rappel.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precision[:-1], label='Precision')
    plt.plot(thresholds, recall[:-1], label='Recall')
    plt.xlabel('Seuil de décision')
    plt.ylabel('Score')
    plt.title('Courbe de Précision-Rappel en fonction du seuil de décision')
    plt.legend()
    plt.savefig(f"{evaluation_dir}/{filename}_precision_recall_curve.png")
    plt.close()


def plot_probability_histogram(y_pred_prob, evaluation_dir, filename):
    """
    Fonction pour générer et sauvegarder l'histogramme des probabilités.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(y_pred_prob, bins=30)
    plt.title('Histogramme des probabilités')
    plt.xlabel('Probabilité')
    plt.ylabel('Fréquence')
    plt.savefig(f"{evaluation_dir}/{filename}_probability_histogram.png")
    plt.close()


def model_evaluation_metrics_and_visualizations(best_estimator, data, timestamp, filename):
    """
    Cette fonction évalue les performances du modèle sur les ensembles d'entraînement
    et de test, puis génère des visualisations et des rapports de classification.
    """
    # Extraire les données du dictionnaire
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    # Créer un dossier pour chaque évaluation, basé sur le timestamp
    evaluation_dir = f"output/evaluation_{timestamp}"
    os.makedirs(evaluation_dir, exist_ok=True)

    # Calcul des rapports de classification
    train_report = classification_report(y_train, best_estimator.predict(x_train),\
        output_dict=False)
    test_report = classification_report(y_test, best_estimator.predict(x_test), output_dict=False)

    # Sauvegarder les rapports de classification
    save_classification_reports(train_report, test_report, evaluation_dir, filename)

    # Calcul des probabilités pour la classe 1 (classe des vrais billets)
    y_pred_prob = best_estimator.predict_proba(x_test)[:, 1]

    # Calcul des courbes de précision et rappel
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

    # Affichage et sauvegarde des courbes et de l'histogramme
    plot_precision_recall_curve(precision, recall, thresholds, evaluation_dir, filename)
    plot_probability_histogram(y_pred_prob, evaluation_dir, filename)

    # Affichage et sauvegarde de la matrice de confusion
    y_pred = best_estimator.predict(x_test)
    plot_confusion_matrix(y_test, y_pred, evaluation_dir, filename)

if __name__ == "__main__":
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
    FILENAME = "logistic_model"  # Nom du fichier

    # Appeler la fonction pour évaluer le modèle et générer les visualisations
    model_evaluation_metrics_and_visualizations(
        best_model,
        df_input,
        actual_timestamp,
        FILENAME
    )
