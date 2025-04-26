import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import joblib

#lecture du fichier
df = pd.read_csv("data/employes_rh.csv")

#separation des donnees en features et target
X = df[["age","anciennete","satisfaction","salaire"]]
y = df["depart"]

#separation en entrainement et test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#création du modèle
model = RandomForestClassifier(n_estimators=300, random_state=42,max_depth=10)


#entrainement du modèle
model.fit(X_train,y_train)

#predictions
y_pred = model.predict(X_test)

#Evaluation

print(classification_report(y_test,y_pred))

#sauvegarde du modèle

joblib.dump(model,"model/model_turn_over.pkl")
