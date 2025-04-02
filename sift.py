import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('01-Data/dnd.jpg')

# SIFT (Scale-Invariant Feature Transform) è un algoritmo in grado di 
# rilevare punti caratteristici robusti (keypoints) indipendenti dalla scala e rotazione.

# Creazione un oggetto SIFT
sift = cv2.SIFT_create()

# Rileva i punti caratteristici (keypoints) e calcola i relativi descrittori per l'immagine.
# I keypoints sono punti rilevanti, tipicamente con geometrie ben definiti.
# I descrittori sono vettori che descrivono l'intorno di ciascun keypoint.
# Il secondo parametro (None) indica che non viene utilizzata una maschera.
keypoints, descriptors = sift.detectAndCompute(img, None)


# Disegna i keypoints sull'immagine.
#   Il parametro img è l'immagine originale.
#   keypoints è la lista dei punti rilevati.
#   Il terzo parametro specifica l'immagine sulla quale disegnare (qui si riutilizza l'immagine originale).
#   (255,0,0) definisce il colore (in BGR) con cui disegnare i keypoints; in questo caso rosso.
#   Il flag cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS permette di disegnare i keypoints con dimensioni e orientamenti che corrispondono alle caratteristiche rilevate.
img_with_keypoints = cv2.drawKeypoints(img, keypoints, img, (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Keypoints', img_with_keypoints)
cv2.waitKey(0)