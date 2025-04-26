#import de pandas pour la manipulation de donnèes
import pandas as pd 
import streamlit as st

#Titre de l'application
st.title('Nettoyage de donnèes RH')

#upload du fichier
upload_file = st.file_uploader("Choisissez un fichier csv",type="csv")
if upload_file is not None:

 
    #lecture du fichier csv
    df = pd.read_csv(upload_file,index_col=None)
    #apperçu des 5 premiers lignes
    st.subheader('apperçu des donnèes')
    st.dataframe(df.head())
    #statistiques simples
    st.subheader('statistiques')
    st.write("Age moyen : ",df.age.mean())
    st.write("Postes les plus fréquents : ",df.poste.value_counts())
    #suppression des doublons
    df_cleaned = df.drop_duplicates()

    st.subheader('Données nettoyées')
    st.dataframe(df_cleaned.head())

    #télécharger le fichier néttoyé
    st.download_button(
        label="Télécharger le fichier nettoyé",
            data = df_cleaned.to_csv(index=False).encode('utf-8'),
        file_name= "employes_nettoyes.csv",
           mime='text/csv'   )   