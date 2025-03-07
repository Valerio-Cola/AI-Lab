import cv2
import numpy as np

# Carica immagine
img = cv2.imread('01-Data/lena.png')

# Proprietà dell'immagine width, height, channels, num pixel
print(img.shape[1], img.shape[0], img.shape[2], img.size) 

# Mostra immagine su finestra
cv2.imshow('Lena', img)

# Attendi pressione di un tasto qualsiasi
cv2.waitKey(0) 
# Chiudi tutte le finestre
cv2.destroyAllWindows() 

# Salva immagine
cv2.imwrite('01-Data/lena_copy.png', img)

# Creazione di un'immagine vuota con numpy mediante una matrice di zeri = pixel neri 512x512x3 (RGB)
empty_img = np.zeros((512, 512, 3), dtype='uint8')
cv2.imshow('Empty Image', empty_img)
cv2.waitKey(0)

# Valori BGR di un pixel in posizione 0,0 
b,g,r = img[0, 0] 
# Cambio colore dei pixel nelle prime 100 righe e prime 100 colonne 
img[:100, :100] = (255, 0, 0) # change the color of the top-left corner to blue
cv2.imwrite('corner.png', img) 

# Crop di un'immagine 
crop = img[100:300, 100:300] 
cv2.imwrite('crop.png', crop)  