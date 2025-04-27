import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import joblib
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


#lecture du fichier
df = pd.read_csv("data/employes_rh.csv")

#separation des donnees en features et target
colonnes_a_garder = [col for col in df.columns if col != 'depart']

X = df[colonnes_a_garder]
y = df["depart"]

#separation en entrainement et test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#création du modèle
#model = RandomForestClassifier(n_estimators=300, random_state=42,max_depth=10)


#Preparation des pipelines

numeric_pipeline = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="mean")),
    ("scaler",StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("impute",SimpleImputer(strategy="most_frequent")),
    ("encode",OneHotEncoder(handle_unknown='ignore'))
])


#creation du preprocesseur

prepocesseur = ColumnTransformer(transformers=[
    ('num',numeric_pipeline,make_column_selector(dtype_include=['int64','float64'])),
    ('cat',categorical_pipeline,make_column_selector(dtype_include=['object']))
])

#Pipeline complet

pipeline = Pipeline(steps=[
    ('preprocessing',prepocesseur),
    ('model',RandomForestClassifier(n_estimators=300, random_state=42,max_depth=10))
])

#entrainement du modèle
pipeline.fit(X_train,y_train)

#predictions
y_pred = pipeline.predict(X_test)

#importance des variables 

# 1. Récupérer les features après preprocessing
feature_names = pipeline.named_steps['preprocessing'].get_feature_names_out()

# 2. Récupérer le modèle
modele = pipeline.named_steps['model']

# 3. Importance des features
importances = modele.feature_importances_

# 4. Tableau d'importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print(feature_importance)


#Evaluation

print(classification_report(y_test,y_pred))


#sauvegarde du modèle

joblib.dump(pipeline,"model/model_turn_over.pkl")
