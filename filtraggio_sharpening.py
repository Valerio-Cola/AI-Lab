import cv2
import numpy as np

img = cv2.imread('01-Data/salt_pepper.png')

# Matrice di convoluzione per il filtraggio
my_kernel = np.array([
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5]
])

# Filtro2D, -1 indica che la profondità dell'immagine è la stessa dell'immagine originale 
filtered_img = cv2.filter2D(img, -1, my_kernel)

# Blurring standard dell'immagine, 7x7 è la dimensione del kernel 
# ovvero la finestra di pixel dei quali si calcola la media
blur_img = cv2.blur(img, (7, 7))

# BoxFilter, uguale a blur ma migliore gestione spigoli
boxfilter_img = cv2.boxFilter(img, -1, (7, 7))

# GaussianBlur, come il blur ma con un kernel gaussiano,
# 0 indica che verrà calcolata automaticamente l'intensità in base alla dimensione del kernel
gaussian_img = cv2.GaussianBlur(img, (7, 7), 0)

# MedianBlur, utile per rimuovere il rumore di tipo salt and pepper
medianblur_img = cv2.medianBlur(img, 3)

cv2.imshow('Original Image', img)
cv2.imshow('Filtered Image', filtered_img)
cv2.imshow('Blurred Image', blur_img)
cv2.imshow('Box Filtered Image', boxfilter_img)
cv2.imshow('Gaussian Image', gaussian_img)
cv2.imshow('Median Blur Image', medianblur_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


img2 = cv2.imread('01-Data/lena.png')

# Unsharp Masking, L'immagine originale viene ponderata con un peso di 1.5,
# mentre quella sfocata con -0.5; il risultato ne evidenzia i bordi.
# Il parametro finale (0) è un valore scalare aggiunto a ciascun pixel.
smoothed_img = cv2.GaussianBlur(img2, (9, 9), 10)
final_img = cv2.addWeighted(img2, 1.5, smoothed_img, -0.5, 0)

# Sharpening, I pixel dell'immagine originale vengono ponderati con un peso di 5 centralmente,
# mentre quelle circostanti con -1/0; il risultato ne evidenzia i bordi.
sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])
sharpen_img = cv2.filter2D(img2, -1, sharpen_kernel)

cv2.imshow('Original Image', img2)
cv2.imshow('Smoothed Image', final_img)
cv2.imshow('Sharpen Image', sharpen_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Carica in modalità scala di grigi 
gray_img = cv2.imread('01-Data/lena.png', cv2.IMREAD_GRAYSCALE)

# Conversione in scala di grigi
img = cv2.imread('01-Data/lena.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calcolo dei gradienti di Sobel
grad_x = cv2.Sobel(gray_img, -1, 1, 0)
grad_y = cv2.Sobel(gray_img, -1, 0, 1)
# Converte i gradienti in valori assoluti formato uint8
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
# Combina i gradienti con pesi uguali
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# Filtro Laplaciano per il rilevamento dei bordi si basa sulla derivata seconda 
dst = cv2.Laplacian(gray_img, -1, ksize=3)
abs_dst = cv2.convertScaleAbs(dst)

cv2.imshow('Original Image', img)
cv2.imshow('Gray Image', gray_img)
cv2.imshow('Gray Image', gray_img)
cv2.imshow('Sobel', grad)
cv2.imshow('Laplacian', abs_dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
