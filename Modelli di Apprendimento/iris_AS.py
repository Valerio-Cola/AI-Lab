from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

# Carica dataset
iris = sns.load_dataset("iris")

# X sono i dati, Y sono le etichette
X_iris = iris.drop("species", axis=1)
y = iris["species"]

# Dividi il dataset in training e test set
# 90% training, 10% testing
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y, train_size=0.9, random_state=42)

# Seleziona il modello
# In questo caso utilizziamo un classificatore Naive Bayes
model = GaussianNB()

# Addestra il modello
model.fit(Xtrain, ytrain)

# Previsione sul test set
# Il modello prevede le etichette per il test set X con i dati
y_pred = model.predict(Xtest)

# Calcola l'accuratezza del modello
# Confronta le etichette previste con quelle reali
accuracy = accuracy_score(ytest, y_pred)
print(f"Accuracy: {accuracy:.2f}")