from sklearn.feature_extraction import DictVectorizer

data = [
    {'price': 100, 'quantity': 200, 'name': 'spam'},
    {'price': 500, 'quantity': 500, 'name': 'eggs'},
    {'price': 200, 'quantity': 100, 'name': 'bacon'}
]

name = {'spam': 0, 'eggs': 1, 'bacon': 2}

# one-hot encoding
vec = DictVectorizer(sparse=False, dtype=int)

X = vec.fit_transform(data)
print(vec.get_feature_names_out())
print(X)