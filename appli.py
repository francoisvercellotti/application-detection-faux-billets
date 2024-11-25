"""
Application de prédiction pour détecter si un billet est authentique ou faux.
Ce script charge un modèle pré-entraîné et effectue des prédictions sur les données des billets.
"""


import pandas as pd
import joblib
import streamlit as st

# Chemin vers le modèle pré-entraîné
MODEL_PATH = "./model/best_model.pkl"

"""
Application de prédiction pour détecter si un billet est authentique ou faux.
Ce script charge un modèle pré-entraîné et effectue des prédictions sur les données des billets.
"""

def load_data(file):
    """
    Charge les données téléversées (CSV ou Excel).

    Args:
        file: Fichier téléversé (CSV ou Excel).

    Returns:
        pd.DataFrame: Données chargées avec les colonnes nécessaires.
    """
    # Vérification du type de fichier
    if file.name.endswith('.csv'):
        data = pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        data = pd.read_excel(file)
    else:
        st.error("Format de fichier non supporté. Veuillez téléverser un fichier CSV ou Excel.")
        return None

    # Sélectionner uniquement les colonnes nécessaires
    columns_needed = ['id', 'diagonal', 'height_left', 'height_right',\
        'margin_low', 'margin_up', 'length']
    data = data[[col for col in columns_needed if col in data.columns]]
    return data

def load_model():
    """
    Charge le modèle pré-entraîné depuis le fichier.

    Returns:
        Le modèle chargé.
    """
    return joblib.load(MODEL_PATH)

def predict_data(model, data):
    """
    Effectue des prédictions sur les données téléversées.

    Args:
        model: Modèle chargé.
        data (pd.DataFrame): Données à prédire.

    Returns:
        pd.DataFrame: Données avec une colonne 'Prediction'.
    """
    # Préparer les données pour la prédiction
    data_for_prediction = data.drop(columns=['id'])

    # Effectuer les prédictions
    predictions = model.predict(data_for_prediction)
    data['Prediction'] = predictions
    data['Prediction'] = data['Prediction'].map({1: 'Vrai', 0: 'Faux'})

    return data

def main():
    st.title("Application de Prédiction pour Billets de Banque")

    st.markdown("""
    **Instructions :**
    1. Téléversez un fichier CSV ou Excel contenant les caractéristiques des billets.
    2. L'application ajoutera une colonne avec la prédiction pour chaque billet (`Vrai` ou `Faux`).
    3. Téléchargez le fichier avec les prédictions.
    """)

    # Téléversement du fichier
    uploaded_file = st.file_uploader("Téléversez votre fichier (CSV ou Excel)",\
        type=['csv', 'xls', 'xlsx'])

    if uploaded_file is not None:
        # Charger les données
        data = load_data(uploaded_file)

        if data is not None:
            st.write("Aperçu des données chargées :")
            st.write(data.head())

            # Charger le modèle
            model = load_model()

            # Effectuer les prédictions
            predicted_data = predict_data(model, data)

            st.write("Prédictions effectuées :")
            st.write(predicted_data.head())

            # Bouton pour télécharger le fichier de résultats
            output_file = uploaded_file.name.split('.')[0] + '_predictions.csv'
            csv_data = predicted_data.to_csv(index=False, sep=';')
            st.download_button(
                label="Télécharger le fichier avec prédictions",
                data=csv_data,
                file_name=output_file,
                mime='text/csv'
            )

if __name__ == "__main__":
    main()
