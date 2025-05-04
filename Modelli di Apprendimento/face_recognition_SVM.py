from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt

faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

# Visualizza le immagini dei volti
fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])
plt.show()

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

# Utilizza PCA per ridurre la dimensionalità delle immagini
# e SVC per la classificazione
# Riduci la dimensionalità delle immagini a 150 componenti principali
# e utilizza SVC con kernel RBF per la classificazione
pca = PCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

# Dividiamo dataset
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, random_state=42)

# Addestriamo il modello con GridSearchCV, che cerca i migliori iperparametri
# per addestrare al meglio il modello SVC
from sklearn.model_selection import GridSearchCV
param_grid = {'svc__C': [1, 5, 10, 50], 'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)
grid.fit(Xtrain, ytrain)
print(grid.best_params_)

model = grid.best_estimator_
yfit = model.predict(Xtest)

# Visualizzazione dei risultati
fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1], color='black' if yfit[i] == ytest[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14)
plt.show()

# Stampa il report di classificazione
#   precision = è la percentuale di veri positivi rispetto al totale dei positivi previsti
#   recall = è la percentuale di veri positivi rispetto al totale dei positivi reali
#   f1-score = è la media armonica tra precision e recall
#   support = è il numero di campioni per ciascuna classe
# La media è calcolata come la media pesata per il numero di campioni in ciascuna classe
from sklearn.metrics import classification_report
print(classification_report(ytest, yfit, target_names=faces.target_names))

# Stampa la matrice di confusione
from sklearn.metrics import confusion_matrix
import seaborn as sns

mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=faces.target_names, yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()