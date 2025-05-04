# Le SVM sono un tipo di algoritmo di apprendimento supervisionato
# che può essere utilizzato per la classificazione o la regressione.
# Le SVM cercano di trovare un iperpiano che separa i dati in classi diverse
# in modo ottimale. L'iperpiano è scelto in modo tale da massimizzare il margine tra le classi.


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from seaborn import set_theme
from sklearn.datasets._samples_generator import make_blobs


X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)

# Simuliamo manualmente il lavoro di una SVM
#Generiamo un array di valori equidistanti tra -1 e 3.5
xfit = np.linspace(-1, 3.5)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

# Disegna le rette che dividono i dati in due classi
# m = coefficiente angolare, b = intercetta, xfit = array di valori x
# La retta è definita dalla formula y = mx + b = m * xfit + b
# -k è il colore della retta (nero)
for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    plt.plot(xfit, m * xfit + b, '-k')

# Imposta il range di visualizzazione dell'asse x
plt.xlim(-1, 3.5)
plt.show()

# Facciamo la stessa cosa ma con un margine di errore
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    # Disegna le linee di margine superiore e inferiore di errore
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA', alpha=0.4)
plt.xlim(-1, 3.5)
plt.show()


# Train del modello SVM
from sklearn.svm import SVC

model = SVC(kernel='linear', C=1E10)
model.fit(X, y)

# Visualizza i dati e il margine di errore
def plot_svc_decision_function(model, ax=None, plot_support=True):

    # se ax è None, crea un nuovo asse
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # crea una griglia di punti per il piano
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # disegna il piano di decisione e i margini
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    # disegna i punti di supporto
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=300, linewidth=1, facecolors='none');
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model)
plt.title('SVM con margine di errore')
plt.show()


# Kernel SVM
from sklearn.datasets._samples_generator import make_circles
from mpl_toolkits import mplot3d
from ipywidgets import interact, fixed

# Genera un dataset circolare, è quindi ovvio che non può essere separato linearmente
X, y = make_circles(100, factor=.1, noise=.1)
clf = SVC(kernel='linear').fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf, plot_support=False)
plt.title('SVM con kernel lineare')
plt.show()

# Ora proviamo a separare i dati con un kernel polinomiale in 3D

def plot_3D(elev=30, azim=30, X=X, y=y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    r = np.exp(-np.sum(X**2, axis=1))
    ax.scatter(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')
interact(plot_3D, elev=[-90, 90], azim=(-180, 180), X=fixed(X), y=fixed(y))
plt.show()

# Notiamo come i dati siano separabili in 3D, in particolare sull'asse r
# Utilizziamo kernel trick per separare i dati in 3D
# Kernel trick: calcola il prodotto scalare in uno spazio ad alta dimensione senza calcolarne le coordinate
# I dati verranno mostrati in 2D e separati da cerchi
clf = SVC(kernel='rbf', C=1E6)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=300, lw=1, facecolors='none')
plt.title('SVM con kernel RBF')
plt.show()


# Tuning dei parametri
# Cosa fare se non sono perfettamente separabili, e ci sono degli overlap?
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=1.2)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plt.title('Dati non separabili')
plt.show()

# Andremo a modificare i parametri C e gamma
# C: penalizza i punti di errore, più è alto più penalizza
# gamma: definisce la distanza di influenza dei punti di supporto, più è alto più influenza i punti vicini
# In questo modo possiamo evitare di avere un margine di errore troppo grande
# e quindi avere un modello più preciso

X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.8)
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

for axi, C in zip(ax, [10.0, 0.1]):
    model = SVC(kernel='linear', C=C).fit(X, y)
    axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(model, axi)
    axi.scatter(model.support_vectors_[:, 0],
    model.support_vectors_[:, 1], s=300, lw=1, facecolors='none')
    axi.set_title('C = {0:.1f}'.format(C), size=14)
plt.show()
