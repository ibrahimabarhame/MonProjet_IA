#import de pandas pour la manipulation de donnèes
import pandas as pd 
import streamlit as st
#lecture du fichier
df = pd.read_csv("data/employes_rh.csv",index_col=None)
#affichage des premieres lignes
print(df.head())

#stat

print("age moyen :",round(df.age.mean()),"ans")
print("Taux de départ : ",round(df.depart.mean()*100),"%")
print("salaire moyen : ",round(df.salaire.mean()),"€")
