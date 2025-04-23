# Handwritten supervised classification
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

digits = load_digits()

# Shape of the images array 1797 samples, 8x8 pixels
print(digits.images.shape)  

# 
fig, axes = plt.subplots(10, 10, figsize=(8,8), subplot_kw={'xticks':[], 'yticks':[]})

# Visualizzazione delle prime 100 immagini
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary')
    ax.text(0.05, 0.05, str(digits.target[i]), color='red', fontsize=8, ha='left', va='top')
plt.show()


X = digits.data
print(X.shape)  # 1797 samples, 64 features (8x8 pixels flattened)

y = digits.target
print(y.shape)  # 1797 labels, una per ogni immagine

# Dividiamo il dataset in un training set e un test set
# 70% training, 30% test
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.7, random_state=42)

# Creiamo un classificatore Naive Bayes
model = GaussianNB()
# Addestriamo il modello con il training set
model.fit(Xtrain, ytrain)
# Prediciamo le etichette del test set per valutare l'accuratezza del modello
y_pred = model.predict(Xtest)

accuracy = accuracy_score(ytest, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Visualizzazione della matrice di confusione, in cui le righe sono le etichette vere e le colonne quelle predette
# La diagonale principale mostra i valori corretti, mentre gli altri valori mostrano gli errori di classificazione
confusion_matr = confusion_matrix(ytest, y_pred)
sns.heatmap(confusion_matr, square=True, annot=True, fmt='d', cmap='Blues', cbar=False) 
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Visualizzazione dei primi 100 test set con le etichette predette
# e il colore verde se corretto, rosso se errato
fig, axes = plt.subplots(10, 10, figsize=(8,8), subplot_kw={'xticks':[], 'yticks':[]})

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary')
    ax.text(0.05, 0.05, str(digits.target[i]), color='green' if (ytest[i] == y_pred[i]) else 'red', fontsize=8, ha='left', va='top')
plt.show()