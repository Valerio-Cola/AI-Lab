# Selezione di un cartellone e inserimento di un'immagine al suo interno

import cv2
import numpy as np

# Funzione per il click del mouse sull'immagine 
def onClick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(dst_points) < 4:
            dst_points.append([x, y])
            cv2.circle(img_copy, (x, y), 30, (0, 255, 0), -1)
            cv2.imshow('Click', img_copy)


img2 = cv2.imread('01-Data/ezio.jpg')
base_img = cv2.imread('01-Data/billboard.jpg')
img_copy = base_img.copy()

# Prendo le dimensioni delle immagini
base_h, base_w = base_img.shape[:2]
img2_h, img2_w = img2.shape[:2]

# Definisco i punti di partenza dell'immagine da inserire
# (in questo caso sono i 4 angoli dell'immagine)
src_float = np.float32([
    [0, 0],
    [0, img2_h],
    [img2_w, img2_h],
    [img2_w, 0]
])

# Definisco i punti di arrivo dell'immagine da inserire
# (in questo caso sono i 4 angoli del cartellone)
dst_points = []

# Finestra per selezionare i punti dove inserire l'immagine con la funzione onClick
cv2.namedWindow('Click', cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback('Click', onClick)
cv2.imshow('Click', base_img)
cv2.waitKey(0)

# Converto i punti in float32 per il calcolo della matrice di trasformazione
dst_float = np.float32(dst_points)
H = cv2.getPerspectiveTransform(src_float, dst_float)

# Applico la trasformazione prospettica all'immagine del poster
warped = cv2.warpPerspective(img2,H,(base_w,base_h))

# Calcolo la maschera per l'immagine da inserire
# La maschera è un'immagine nera, grande quanto l'immagine dove ho selezionato i punti
mask = np.zeros(base_img.shape, dtype=np.uint8)

# Disegno un poligono bianco (255, 255, 255) sulla maschera
# che ha come vertici i punti di arrivo (dst_float)
cv2.fillConvexPoly(mask, np.int32(dst_float), (255, 255, 255))

# La maschera è un'immagine nera, quindi per inserire l'immagine da inserire 
# devo fare un bitwise_and tra la maschera e l'immagine da inserire
# e poi fare un bitwise_or tra il risultato e l'immagine di base
# In questo modo l'immagine da inserire viene inserita solo nei punti bianchi della maschera
mask = cv2.bitwise_not(mask)
masked_billboard = cv2.bitwise_and(base_img, mask)
final_billboard = cv2.bitwise_or(masked_billboard, warped)

# Mostro l'immagine finale
cv2.namedWindow('Warped Image', cv2.WINDOW_KEEPRATIO)
cv2.imshow('Warped Image', final_billboard)
cv2.waitKey(0)
cv2.destroyAllWindows()