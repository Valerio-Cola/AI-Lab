import cv2
import matplotlib.pyplot as plt

img = cv2.imread('01-Data/lena.png')
color = ('b', 'g', 'r')

# Per ogni canale/colore:
for i, col in enumerate(color):
    # Calcola l'istogramma del canale corrente.
    # cv2.calcHist accetta l'immagine 'img', il canale corrente [i], nessuna maschera (None),
    # il numero di bins [256] per l'istogramma e l'intervallo [0, 256] dei valori di intensità.
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    
    # Esegue il plot dei valori dell'istogramma per il canale corrente,
    # usando il colore specificato ('b' per blu, 'g' per verde, 'r' per rosso).
    plt.plot(hist, color=col)

# Imposta il limite sull'asse x del grafico da 0 a 256, per coprire tutte le intensità possibili.
plt.xlim([0, 256])
plt.show()