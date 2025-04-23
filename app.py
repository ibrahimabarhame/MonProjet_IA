#import de pandas pour la manipulation de donnèes
import pandas as pd 
#lecture du fichier csv
df = pd.read_csv("data/employes.csv")
#apperçu des 5 premiers lignes
print(df.head())

#statistiques simples
df.describe()

#oubien
print(f'age moyen est de {df["age"].mean()}')
print(f"Les postes les plus frèquents sont : {df['poste'].value_counts()}")

#suppression des doublons
df = df.drop_duplicates()

#export du fichier nettoyè
df.to_csv("fichier_nettoyé.csv",index=False)