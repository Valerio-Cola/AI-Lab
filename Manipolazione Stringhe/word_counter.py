from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
s = [
        'problem of evil',
        'evil queen',
        'horizon problem'
]

vec = CountVectorizer()
X = vec.fit_transform(s)
print(vec.get_feature_names_out(), X)

x_pd = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())
print(x_pd)


vec = TfidfVectorizer()
X = vec.fit_transform(s)
print(vec.get_feature_names_out(), X)

x_pd = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())
print(x_pd)

