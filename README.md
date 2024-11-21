# Application de détection de faux billets

Dans ce projet, nous développons un modèle de détection de faux billets à partir d'un jeu de données comprenant 1000 vrais billets et 500 faux billets. Après une analyse exploratoire des données et un prétraitement pour les optimiser, nous testerons plusieurs modèles de machine learning afin de sélectionner les plus performants.

L'objectif principal est d'obtenir un modèle capable de détecter les faux billets avec un rappel (recall) d'au moins 99%, tout en maintenant une précision supérieure à 98%. Nous procéderons également à l'optimisation des hyperparamètres et à l'évaluation rigoureuse des modèles avant de les intégrer dans une application de détection fiable.

## Etapes
### 1. Exploration des données (EDA)
- Analyse statistique des variables
- Visualisation des distributions des données
- Identification des variables discriminantes

### 2. Prétraitement des données
- Gestion des valeurs manquantes
- Normalisation et standardisation des données
- Sélection des caractéristiques pertinentes
- Création de nouvelles variables si nécessaire

### 3. Modélisation et évaluation
- Comparaison des performances de différents modèles de machine learning (régression, arbres de décision, forêts aléatoires, KNN, etc.)
- Sélection du meilleur modèle en fonction de critères de performance (précision, rappel, F1-score)
- Recherche des meilleurs hyperparamètres à l'aide de techniques comme la recherche par grille (GridSearch)

### 4. Optimisation et intégration
- Optimisation des hyperparamètres du modèle sélectionné
- Evaluation du modèle final sur un jeu de test
- Intégration dans une application de détection de faux billets

## Environnement de développement

Le projet a été développé avec **VSCode** et utilise un environnement virtuel pour isoler les dépendances. Vous pouvez configurer votre propre environnement de travail en suivant ces étapes :



├── data/                  # Données d'entrée et sorties
├── model/                 # Modèles entraînés
├── notebooks/             # Notebooks Jupyter pour l'analyse et la modélisation
├── output/                # Résultats et figures de l'analyse
├── scripts/               # Scripts Python pour prétraitement, modélisation, etc.
├── requirements.txt       # Liste des dépendances
├── .gitignore             # Fichiers à ignorer dans Git
└── README.md              # Ce fichier
