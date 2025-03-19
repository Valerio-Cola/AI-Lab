import cv2
import numpy as np

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

