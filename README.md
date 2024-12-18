# Application de détection de faux billets

Dans ce projet, mon entreprise vient de décrocher un contrat avec l’Organisation nationale de lutte contre le faux-monnayage (ONCFM) dans le but de créer une application de détection de faux billets.
Je joue donc ici le rôle d'un senior data analyst envoyé pour mener à bien cette mission.

## Contexte

L'ONCFM est chargée de détecter et de prévenir la circulation de billets contrefaits, mais les processus actuels sont longs et nécessitent une automatisation pour être plus efficaces.

## Objectifs

- Développer une application de machine learning permettant de prédire la nature d'un billet (vrai ou faux) à partir de ses caractéristiques physiques.
- Évaluer et comparer plusieurs algorithmes afin de déterminer le modèle le plus performant pour cette tâche.
- Intégrer le modèle final dans une application fonctionnelle qui pourra être utilisée par les équipes de l'ONCFM sur le terrain.

## Étapes

### 1. Exploration des données (EDA)
- Analyse statistique des variables.
- Visualisation des distributions des données.
- Identification des variables discriminantes.

### 2. Prétraitement des données
- Gestion des valeurs manquantes.
- Encodage et standardisation des données.
- Création de nouvelles variables par transformations polynomiales.
- Sélection des caractéristiques pertinentes.

### 3. Modélisation et évaluation
- Comparaison des performances de différents modèles :
  - Un modèle de **bagging** : `RandomForestClassifier`.
  - Un modèle de **boosting** : `AdaBoostClassifier`.
  - Un modèle de **régression** : `LogisticRegression`.
  - Un modèle de **SVM** : `SVC`.
  - Un modèle de **base** : `KNN`.
- Sélection du meilleur modèle en fonction des critères de performance (précision, rappel, F1-score).
- Recherche des meilleurs hyperparamètres à l'aide de techniques comme la recherche par grille (`GridSearchCV`) ou la recherche aléatoire (`RandomSearchCV`).

### 4. Optimisation et intégration
- Optimisation des hyperparamètres du modèle sélectionné.
- Évaluation du modèle final sur un jeu de test.
- Intégration dans une application de détection de faux billets.

### 5. Support de présentation
- Une présentation structurée détaillant :
  - Les traitements réalisés.
  - Les modèles évalués.
  - Le choix du modèle final.

<img src="https://github.com/user-attachments/assets/79760fe8-e125-499b-8c08-c6102a7d8988" alt="image" width="600">

<img src="https://github.com/user-attachments/assets/ee8bbe98-2b36-42c3-a8f9-ba39a66ed0cd" alt="image" width="600">

<img src="https://github.com/user-attachments/assets/69e7ec7b-c6d9-4f2b-ba17-ae4402843092" alt="image" width="600">



## Pour aller plus loin : Mise en production et bonnes pratiques de développement

Les étapes 1 à 4 ont été réalisé sur un notebook, pour aller plus loin et afin de garantir la robustesse, la qualité et la portabilité de l'application, des outils et méthodes avancés ont été utilisés dans une optique de mise en production.

Cette partie a été développé avec **VSCode** et en utilisant wsl2



## Etapes additionnelles

### 1. Contrôle de version avec Git et GitHub
- Suivi des versions avec Git pour gérer efficacement les modifications.
- Hébergement du projet sur GitHub pour faciliter la collaboration.
- Structuration des branches :



Le projet est organisé en plusieurs répertoires et fichiers pour garantir une meilleure structuration et lisibilité du code.

### Répertoires principaux

 `data/`  
  Données d'entrée et de sortie.  
 

 `model/`  
  Modèles entraînés.  
  

`notebooks/`  
  Notebooks Jupyter pour l'analyse et la modélisation.  
  

 `output/`  
  Résultats et figures de l'analyse.  
 

 `scripts/`  
  Scripts Python pour le prétraitement, la modélisation, etc.  
  

### Fichiers principaux

`.dockerignore`  
  Définit les fichiers à exclure lors de la construction du conteneur Docker.  
  
`.gitignore`  
  Spécifie les fichiers à ignorer dans Git.  
  

 `Dockerfile`  
  Contient les instructions pour construire l'image Docker de l'application.  
  

`Projet n°12.code_workspace.code-workspace`  
  Configuration du workspace pour VSCode.  
  

 `README.md`  
  Document décrivant le projet et ses étapes.  
  

 `appli.py`  
  Script principal de l'application, incluant la fonction de prédiction.  
 

 `requirements.txt`  
  Liste des dépendances nécessaires à l'exécution du projet.  
  



### 2. Structure modulaire et documentation
- Découpage en scripts Python modulaires.
- Documentation rigoureuse avec **Docstrings** pour chaque fonction et module.

### 3. Gestion des environnements
- Utilisation d’un environnement virtuel `venv` pour isoler les dépendances.
- Fichier `requirements.txt` pour une installation simplifiée.

### 4. Contrôle de la qualité du code
- Utilisation de **pylint** pour garantir un code propre et conforme aux bonnes pratiques.
- Correction des erreurs et amélioration des performances du code grâce aux retours du linter.

### 5. Mise en place de tests unitaires
- Tests unitaires avec **pytest** pour valider les fonctions critiques.

### 6. Conteneurisation avec Docker
- Création d’un fichier **Dockerfile** pour garantir la portabilité de l'application.
- Utilisation d'un Dockerfile pour simplifier le déploiement.

### 7. Déploiement d’une application web
- Développement d’une interface utilisateur avec **Streamlit**.
- Fonctionnalités principales :
- Téléchargement de nouveaux fichiers pour prédiction.
- Affichage des résultats des modèles en temps réel.
---

## Compétences mobilisées
- **Analyse de données** : Exploration et visualisation des données (EDA).
- **Machine Learning** : Prétraitement des données, modélisation, et évaluation de modèles.
- **Développement logiciel** : Programmation en Python, structuration modulaire, et documentation.
- **DevOps** : Contrôle de version avec Git, conteneurisation avec Docker, gestion des environnements virtuels.
- **Déploiement d'application** : Développement d'interfaces utilisateur avec Streamlit, intégration de modèles.

## Outils utilisés
- **Python** : Pandas, Scikit-learn, Matplotlib, Seaborn, Streamlit, pytest.
- **Environnements** : VSCode, WSL2, virtualenv.
- **Gestion de version** : Git, GitHub.
- **Linter** : pylint.
- **Conteneurisation** : Docker.
- **Optimisation de modèles** : GridSearchCV, RandomSearchCV.

## Conclusion

Ce projet représente une étape clé dans ma formation en tant que data analyst. Il m'a permis de mettre en pratique des concepts fondamentaux du machine learning, tels que l'analyse exploratoire, le prétraitement des données, la modélisation et l'évaluation des performances. En travaillant sur une problématique concrète, j'ai également renforcé mes compétences en structuration de projet, gestion de version avec Git, et mise en production grâce à des outils comme Docker.

L'aspect multidisciplinaire du projet, combinant analyse des données, programmation et déploiement d'application, m'a permis d’acquérir une vision globale du cycle de vie d’un projet de data science. Ce type de projet me prépare non seulement à des défis techniques, mais aussi à répondre aux besoins réels des entreprises dans un contexte professionnel.

En résumé, ce projet constitue un excellent tremplin pour développer des compétences pratiques et essentielles, tout en consolidant mes connaissances théoriques dans un cadre appliqué.

