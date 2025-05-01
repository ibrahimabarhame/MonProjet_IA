#import de pandas pour la manipulation de donnèes
import pandas as pd 
import streamlit as st
import joblib

#affichage des premieres lignes
model = joblib.load('model/model_turn_over.pkl')

st.title("Prédiction du risque de départ des employés")

upload_file = st.file_uploader("choisissez vottre fichier",type="csv")

if upload_file is not None :

    df = pd.read_csv(upload_file,index_col=None)
    st.subheader("Apperçu des donnèes")
    st.dataframe(df.head())

    #preparation des donnèes pour prediction

    colonnes_a_garder = [col for col in df.columns if col != 'depart']

    X = df[colonnes_a_garder]

    predictions = model.predict(X)

    df["Risque_depart"] = predictions

    st.subheader("Prediction")
    st.dataframe(df)

    st.download_button(
        label="Télécharger le fichier avec prédictions",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='employes_avec_risque.csv',
        mime='text/csv'
    )