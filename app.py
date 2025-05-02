#import de pandas pour la manipulation de donnèes
import pandas as pd 
import streamlit as st
import joblib

#affichage des premieres lignes
model = joblib.load('model/model_turn_over.pkl')

st.title("Prédiction du risque de départ des employés")

st.info("⚠️ Ceci est une version de démonstration utilisant un modèle IA simple sur des données fictives. Le score de risque est à but illustratif.")

colonnes_attendues = ["id","age","anciennete","satisfaction","salaire","promotions_recues","annees_depuis_derniere_promotion","jours_absence","niveau_etude","type_poste","type_contrat","departement"]

st.markdown("**📋 Format attendu :**")
st.write(colonnes_attendues)


upload_file = st.file_uploader("choisissez vottre fichier",type="csv")

if upload_file is not None :

    df = pd.read_csv(upload_file,index_col=None)
    st.subheader("Apperçu des donnèes")
    st.dataframe(df.head())

    #preparation des donnèes pour prediction

    colonnes_a_garder = [col for col in df.columns if col != 'depart']

    #st.write("voici les colonnes à garder dans votre fichier",colonnes_a_garder[1:])

    X = df[colonnes_a_garder]

    predictions = model.predict_proba(X)

    df["score_risque_depart %"] = predictions[:,1]
    df["score_risque_depart %"] = df["score_risque_depart %"].round(2)*100

    st.subheader("Prediction")
    st.dataframe(df)

    st.download_button(
        label="Télécharger le fichier avec prédictions",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='employes_avec_risque.csv',
        mime='text/csv'
    )