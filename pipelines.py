import numpy as np
import matplotlib.pyplot as plt

# Esempio di utilizzo di np per plottare possibili dati
x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 2, 1, 3, 7])
plt.scatter(x, y)
plt.title('Scatter Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

from sklearn.linear_model import LinearRegression

# In questo modo possiamo allenare un modello per ottenere una
# retta che si adatti ai dati e che rappresenta i dati ottimali/media  
X = x[:, np.newaxis]
model = LinearRegression().fit(X, y)
yfit = model.predict(X)
plt.scatter(x, y)
plt.title('Linear Regression')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.plot(x, yfit)
plt.show()


from sklearn.preprocessing import PolynomialFeatures

# Con PolynomialFeatures possiamo generare un polinomio di grado 3
# che si adatti ai dati, in questo modo possiamo vedere come la retta
# sia più vicina ai dati rispetto alla retta di prima
poly = PolynomialFeatures(degree=3, include_bias=False)
X2 = poly.fit_transform(X)
print(X2)

model = LinearRegression().fit(X2, y)
yfit = model.predict(X2)
plt.scatter(x, y)
plt.title('Polynomial Regression')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.plot(x, yfit)
plt.show()

from sklearn.pipeline import make_pipeline
# Infine con make_pipeline possiamo unire i due modelli in uno solo
# e fare il fit direttamente su di esso
model = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
model.fit(X, y)
yfit = model.predict(X)
plt.scatter(x, y)
plt.title('Pipeline Polynomial Regression')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.plot(x, yfit)
plt.show()