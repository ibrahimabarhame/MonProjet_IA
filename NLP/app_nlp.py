#Préparation d'une app qui permet de rapprocher des requêtes et des textes
import os
#lecture du fichier contenant le corpus
chemin_general = os.path.dirname(__file__)
chemin_fichier = os.path.join(chemin_general,"corpus_rh.txt")

with open(chemin_fichier,"r",encoding="utf-8") as fichier :
    lignes = fichier.readlines()
    corpus = [ligne.strip() for ligne in lignes if ligne.strip()] #permet d'eviter les \n et les lignes vides

print(corpus)

#rendre minuscule le corpus

corpus_lower = [element.lower() for element in corpus]
print(corpus_lower)

#suppresion des regex

import re

corpus_nettoyer_1 = [re.sub(r"[^\w\s]", " ",element) for element in corpus_lower]
print(corpus_nettoyer_1)

#tokenisation et stop words en même temps car plusieurs lignes à traiter en plusieur documents
import nltk
nltk.download("punkt")
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download("stopwords")
import spacy
nlp = spacy.load("fr_core_news_sm")

corpus_nettoyer_finale = []
for ligne in corpus_nettoyer_1 :
    tokens = word_tokenize(ligne,language="french")
    tokens_utiles = [element for element in tokens if element not in stopwords.words("french") and element.isalpha()]
    #lemmatiser 
    doc = nlp(" ".join(tokens_utiles))
    lemmes = [mot.lemma_ for mot in doc if mot.lemma_ != ""]
    texte_propre = " ".join(lemmes)
    corpus_nettoyer_finale.append(texte_propre)

#print(corpus_nettoyer_finale)

#vectorisation
