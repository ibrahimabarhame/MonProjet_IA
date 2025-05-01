import pandas as pd 
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
import joblib
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import json 


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

#entrainement du modèle avec mlflow

experiments = [
    {"n_estimators":100,"max_depth":5},
    {"n_estimators":300,"max_depth":10},
    {"n_estimators":500,"max_depth":15}
]

#démarrage de la boucle d'experience
mlflow.set_experiment("turnover_prediction") 

for params in experiments :

    with mlflow.start_run(run_name="Random Forest"):

        pipeline = Pipeline(steps=[
            ('preprocessing',prepocesseur),
            ('model',RandomForestClassifier(n_estimators=params['n_estimators'], random_state=42,max_depth=params['max_depth']))
        ])



        pipeline.fit(X_train,y_train)

        #predictions
        y_pred = pipeline.predict(X_test)

        #evaluation
        accuracy = accuracy_score(y_test,y_pred)
        rec = recall_score(y_test,y_pred)
        prec = precision_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)

        #Log des hyperparametres
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        #Log des metrique
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_dict(classification_report(y_test,y_pred,output_dict=True), "classification_report.json")

        #Log du modèle
        mlflow.sklearn.log_model(pipeline,"model")

