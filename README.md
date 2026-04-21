# DataDoctor AI - AutoML Platform

**DataDoctor AI** est une plateforme automatisée pour l'apprentissage automatique (AutoML). Elle facilite le processus de création de modèles prédictifs, en automatisant le prétraitement des données, la sélection des modèles, l'évaluation et la génération de rapports. Ce projet est conçu pour les utilisateurs ayant peu de connaissances techniques, tout en offrant des résultats robustes et des visualisations détaillées.

## Fonctionnalités

### 1. **Chargement de données**
   - **Support des formats de fichiers** : CSV, Excel
   - **Gestion des erreurs de format** : détection automatique de l'encodage et du séparateur dans les fichiers CSV.
   - **Prévisualisation des données** : affichage des premières lignes du jeu de données et des statistiques de base sur les colonnes.

### 2. **Prétraitement des données**
   - **Gestion des valeurs manquantes** : imputation des valeurs manquantes dans les variables numériques et catégorielles.
   - **Encodage des variables catégorielles** : encodage avec `OneHotEncoder`.
   - **Normalisation des variables numériques** : standardisation via `StandardScaler`.
   - **Suppression des colonnes à forte cardinalité** : suppression automatique des variables catégorielles ayant plus de 50 valeurs uniques.

### 3. **Sélection du modèle**
   - **Tâche automatique** : détection automatique de la tâche (classification ou régression).
   - **GridSearchCV** : optimisation des hyperparamètres pour plusieurs modèles via validation croisée.
   - **Modèles disponibles** :
     - Classification : `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, `XGBoost`, `LightGBM`
     - Régression : `Ridge`, `RandomForestRegressor`, `GradientBoostingRegressor`, `XGBoost`, `LightGBM`
   
### 4. **Évaluation du modèle**
   - **Validation croisée** : utilisation de la validation croisée pour l’évaluation des modèles.
   - **Évaluation sur jeu de test** : calcul des métriques (accuracy, F1-score, AUC-ROC pour la classification, R², MAE, RMSE pour la régression).
   - **SHAP (Shapley Additive Explanations)** : analyse de l'importance des features pour le modèle choisi.
   - **Rapports détaillés** : génération d'un rapport détaillé du modèle, incluant les hyperparamètres optimaux et les performances sur le jeu de test.

### 5. **Génération de rapports**
   - **Rapport LLM** : génération d'un rapport automatisé en français via l'API OpenRouter.
   - **Export en PDF** : possibilité d'exporter le rapport au format PDF.
   
### 6. **Interface Streamlit**
   - **Interface web interactive** pour télécharger des datasets, ajuster les paramètres, visualiser les résultats et télécharger le rapport.

## Prérequis

Avant de commencer, assurez-vous d'avoir les outils suivants installés :

- **Python 3.8+**
- **Git** (pour cloner le repository)
- **Un environnement virtuel** (recommandé pour éviter les conflits de dépendances)

## Installation

### Étape 1 : Cloner le repository

Clonez le repository sur votre machine locale :

```bash
git clone https://github.com/ton_utilisateur/datadoctor-ai.git
cd datadoctor-ai
