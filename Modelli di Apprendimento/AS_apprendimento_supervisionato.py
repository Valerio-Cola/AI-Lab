# Handwritten supervised classification
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

digits = load_digits()

# Shape of the images array 1797 samples, 8x8 pixels
print(digits.images.shape)  

X = digits.data
print(X.shape)  # 1797 samples, 64 features (8x8 pixels flattened)

y = digits.target
print(y.shape)  # 1797 labels, una per ogni immagine


# Creiamo un array lungo quanto l'intero dataset, verrà successivamente diviso in idx_train/test 
# In questo modo è possibile tenere traccia degli indici originali dei dati che dovranno essere associati al y_test/pred
indices = np.arange(X.shape[0])

# Dividiamo il dataset in un training set e un test set
# 70% training, 30% test
Xtrain, Xtest, ytrain, ytest, idx_train, idx_test = train_test_split(X, y, indices, train_size=0.8, random_state=42)

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


# Visualizzazione dei primi 100 test set con le etichette originali e il colore verde se corretto, rosso se errato

# Crea un array di 10 x 10 elementi per visualizzare le immagini
fig, axes = plt.subplots(10, 10, figsize=(8,8), subplot_kw={'xticks':[], 'yticks':[]})

# axes.flat permette di iterare su tutti gli assi in modo sequenziale
for i, ax in enumerate(axes.flat):
    # Grazie a idx_test immagini e relative etichette sono associate correttamente al valore predetto
    ax.imshow(digits.images[idx_test[i]], cmap='binary')
    ax.text(0.5, 0.05, str(digits.target[idx_test[i]]), color='green' if (ytest[i] == y_pred[i]) else 'red', fontsize=8, ha='right', va='top')
    ax.text(0.05, 0.05, f"Pred: {y_pred[i]}", color='red', fontsize=8, ha='left', va='bottom')


# Oppure posso sfruttare Xtest e ytest
fig, axes = plt.subplots(10, 10, figsize=(8,8), subplot_kw={'xticks':[], 'yticks':[]})
for i, ax in enumerate(axes.flat):
    ax.imshow(Xtest[i].reshape(8, 8), cmap='binary')
    ax.text(0.5, 0.05, str(ytest[i]), color='green' if (ytest[i] == y_pred[i]) else 'red', fontsize=8, ha='right', va='top')
    ax.text(0.05, 0.05, f"Pred: {y_pred[i]}", color='red', fontsize=8, ha='left', va='bottom')

plt.show()

