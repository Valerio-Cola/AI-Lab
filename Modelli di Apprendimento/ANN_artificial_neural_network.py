# Implementiamo una Artificial Neural Network (ANN)
# La ANN è composta da più strati di neuroni, ciascuno dei quali è connesso a tutti i neuroni dello strato successivo.

from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Loading the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# Standaedizzazione dei dati
# La standardizzazione è una tecnica di pre-elaborazione dei dati che consiste nel ridimensionare le caratteristiche
# in modo che abbiano media 0 e deviazione standard 1. Questo è particolarmente importante per le reti neurali,
# poiché le caratteristiche con scale diverse possono influenzare negativamente l'addestramento del modello. 
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Inizializzazione del modello, definendo la struttura della rete neurale.
# In questo caso, abbiamo 1 input layer, 2 hidden layers con 5 neuroni ciascuno e 1 output layer.
model = MLPClassifier(hidden_layer_sizes=(5,5,))

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_pred= y_pred, y_true=y_test)

print("Accuracy: ", accuracy)