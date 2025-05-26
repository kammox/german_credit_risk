# german_credit_risk

# 1. Description du projet
Ce projet de machine learning complet utilise un modèle Random Forest pour prédire le risque de crédit des clients d'une banque allemande. Le but est de classer les clients en "risque élevé" (1) ou "risque faible" (0) à partir de données étiquetées. Le projet suit une démarche professionnelle avec modularisation, automatisation des pipelines d’entraînement et de prédiction, gestion des environnements virtuels, gestion des erreurs, logging, et documentation. Streamlit a été utilisé pour rendre le modèle accessible via une interface.

# 2. Technologies utilisées
Python (Pandas, Numpy, Scikit-Learn, Category-Encoders, Scikit-Optimize, XGBoost), Jupyter Notebook, Git/Github, Anaconda, Visual Studio Code.

# 3. Problématique métier
banque souhaite anticiper le risque de crédit pour mieux gérer les risques financiers, optimiser la rentabilité, réduire les pertes, assurer la conformité et segmenter les clients. Le projet vise à identifier au mieux les clients présentant un risque de défaut.

# 4. Insights clés

Les jeunes clients présentent un risque plus élevé.

Les montants de crédit élevés et la durée longue augmentent le risque.

Les clients avec peu d’épargne ou de compte courant présentent un risque accru.

Certains motifs de crédit (vacances, éducation) sont associés à un risque plus élevé.

Les emplois qualifiés et le logement gratuit sont corrélés à des crédits plus importants.

# 5. Modélisation
Prétraitement avec encodage ordinal et ciblé, normalisation par StandardScaler. Random Forest a été choisi pour son potentiel d'amélioration, malgré un léger sur-apprentissage, régularisé via réglage des hyperparamètres par recherche bayésienne. Le seuil de classification a été ajusté pour maximiser le rappel (80%) tout en maintenant une précision acceptable.

# 6. Streamlit
Une interface qui permet d’obtenir des prédictions en temps réel. Le projet sera déployé prochainement sur un cloud (ex : AWS Elastic Beanstalk).

# 7. Dataset
Données issues de la base UCI via Kaggle :
https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk
