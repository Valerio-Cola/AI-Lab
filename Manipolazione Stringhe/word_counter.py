from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Dato un array di stringhe 
s = [
        'problem of evil',
        'evil queen',
        'horizon problem'
]

# Calcolo la matrice di frequenza delle parole
# e la matrice di frequenza delle parole normalizzata con due metodi diversi

# 1. CountVectorizer: conta le occorrenze delle parole
vec = CountVectorizer()
X = vec.fit_transform(s)
print(vec.get_feature_names_out(), X)

x_pd = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())
print(x_pd)

# 2. TfidfVectorizer: calcola la frequenza delle parole normalizzata
vec = TfidfVectorizer()
X = vec.fit_transform(s)
print(vec.get_feature_names_out(), X)

x_pd = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())
print(x_pd)