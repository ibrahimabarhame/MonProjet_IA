
# 🧠 Fiche Mémo – Pipeline NLP Classique (sans TF-IDF)

## 🔁 Étapes du pipeline

### 1️⃣ Lecture du texte
```python
with open("fichier.txt", "r", encoding="utf-8") as f:
    texte = f.read()
```

### 2️⃣ Mise en minuscules
```python
texte = texte.lower()
```

### 3️⃣ Suppression de la ponctuation
```python
import re
texte = re.sub(r"[^\w\s]", " ", texte)
```

### 4️⃣ Tokenisation
```python
from nltk.tokenize import word_tokenize
nltk.download('punkt')  # une seule fois
tokens = word_tokenize(texte, language='french')
```

### 5️⃣ Suppression des stopwords
```python
from nltk.corpus import stopwords
nltk.download('stopwords')  # une seule fois
mots_utiles = [mot for mot in tokens if mot not in stopwords.words('french') and mot.isalpha()]
```

### 6️⃣ Lemmatisation (avec SpaCy)
```python
import spacy
nlp = spacy.load("fr_core_news_sm")
doc = nlp(" ".join(mots_utiles))
lemmes = [token.lemma_ for token in doc if token.lemma_ != ""]
```

## ✅ Résultat final : liste propre de lemmes
Exemple :
```python
['rechercher', 'analyste', 'rh', 'analyser', 'indicateur', 'performance']
```

---

## 📌 À retenir
- `re` → nettoyage avec regex
- `nltk.tokenize` → découpe en mots
- `stopwords` → filtre les mots vides
- `spacy` → lemmatisation pro en français
