from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


"""
I clusters sono gruppi di dati che presentano somiglianze secondo specifici criteri o metriche. In altre parole, l'algoritmo di clustering (come KMeans usato nel tuo codice) cerca di dividere i dati in gruppi in modo che i dati all'interno di ciascun gruppo siano simili tra loro, mentre i dati appartenenti a gruppi diversi abbiano caratteristiche differenti. Questo approccio è tipico dell'apprendimento non supervisionato, in cui non disponi di etichette predefinite.

Nel contesto del tuo script:

L'algoritmo KMeans raggruppa le osservazioni del dataset Iris in 4 cluster, basandosi sulle caratteristiche quantitative dei fiori.
Dopo il fit e il predict, ad ogni osservazione viene assegnata un'etichetta corrispondente al cluster a cui appartiene.
I centroidi dei cluster, calcolati come il punto medio di ognuno, rappresentano il centro di ciascun gruppo.

In sintesi ogni cluster rappresenta un etichetta che identifica un gruppo di dati simili. 
Differentemente dall'apprendimento supervisionato AS, in cui le etichette sono già note e il modello cerca di prevederle.
"""

iris = sns.load_dataset("iris")
X_iris = iris.drop("species", axis=1)
y = iris["species"]

model = KMeans(n_clusters=4) 

model.fit(X_iris)

y_pred = model.predict(X_iris)


plt.scatter(X_iris["sepal_length"], X_iris["sepal_width"], c=y_pred)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("KMeans Clustering of Iris Dataset")
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], s=200, c='red', marker='X', label='Centroids')
plt.show()

plt.scatter(X_iris["petal_length"], X_iris["petal_width"], c=y_pred)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("KMeans Clustering of Iris Dataset")
plt.scatter(model.cluster_centers_[:,2], model.cluster_centers_[:,3], s=200, c='red', marker='X', label='Centroids')
plt.show()