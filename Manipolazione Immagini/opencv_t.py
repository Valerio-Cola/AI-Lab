import cv2
import numpy as np

# Carica immagine
img = cv2.imread('01-Data/lena.png')

# Proprietà dell'immagine width, height, channels, num pixel
print(img.shape[1], img.shape[0], img.shape[2], img.size) 

# Mostra immagine su finestra con dimensioni originali
cv2.namedWindow('Lena', cv2.WINDOW_KEEPRATIO)
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


# Disegnare su un'immagine
canvas = np.zeros((300, 300, 3), dtype='uint8')

green = (0, 255, 0)
rosso = (0, 0, 255)

# Disegna una linea verde dal punto (0, 0) a (300, 300)   
spessore = 5
cv2.line(canvas, (0, 0), (300, 300), green, spessore)

# Rettangolo rosso
spessore = 3 # -1 riempie 
cv2.rectangle(canvas, (10, 10), (60, 60), rosso, spessore)

# Cerchio verde 
raggio = 20
cv2.circle(canvas, (200, 50), raggio, green, spessore)

# Cerchio centrato
x = canvas.shape[1] // 2
y = canvas.shape[0] // 2
cv2.circle(canvas, (x, y), raggio, rosso, -1)

cv2.imshow('Canvas', canvas)
cv2.waitKey(0)


b,g,r = cv2.split(img)
channels = np.hstack([b, g, r])
cv2.imshow('Splitted Channels', channels)

b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]
channels = np.hstack([b, g, r])
cv2.imshow('Splitted Channels', channels)

new_img = cv2.merge([b, g, r])
cv2.imshow('Merged Channels', new_img)
cv2.waitKey(0)

# Resize image 
img_scale_size = cv2.resize(img, (200, 200), interpolation=cv2.INTER_LINEAR)
img_scale_f = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

cv2.imshow('Scaled', img_scale_size)
cv2.imshow('Scaled', img_scale_f)
cv2.waitKey(0)

# Matrice di trasformazione per spostare l'immagine di 200 pixel a destra e 50 pixel in basso
M = np.float32([[1, 0, 200], 
                [0, 1, 50]])
shifted = cv2.warpAffine(img, M, (img.shape[1]*2, img.shape[0]*2))
cv2.imshow('Shifted Image', shifted)

# Trasformazione in scala di grigi
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Image', gray)
cv2.waitKey(0)
