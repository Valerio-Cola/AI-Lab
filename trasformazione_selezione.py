import cv2
import numpy as np


# Funzione che viene chiamata ogni volta che viene registrato un evento sull'immagine
def onClick(event,x,y,flags,params):

    # Considera solo il pulsante sinistro del mouse
    if event == cv2.EVENT_LBUTTONDOWN:
        # Solo 4 punti cliccabili, sono necessari per la perspective transformation 
        if len(src_points) < 4:
            # aggiungiamo il punto che è stato cliccato e disegnamolo, ovviamente bisogna aggiornare l'immagine
            src_points.append([x, y])
            cv2.circle(img_copy, (x,y), 5, (0, 0, 255), 10)
            cv2.imshow('Img', img_copy)

# Mostra l'immagine originale
img = cv2.imread("01-Data/gerry.png")
cv2.namedWindow('Original', cv2.WINDOW_KEEPRATIO)
cv2.imshow("Original", img)

# creo una copia da modificare
img_copy = img.copy()  

# Punti iniziali, l'array non è np, verrà convertito in np.float32 successivamente
src_points = []

# Punti di destinazione
dst_points = np.float32((
    [0,0],
    [0,800],
    [600,800],
    [600,0]
))

# Mostro l'immagine con i click attivi
cv2.namedWindow('Img', cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback('Img', onClick)  
cv2.imshow('Img', img_copy)
cv2.waitKey(0)

# Utilizzo la homography matrix con i punti selezionati per trasformare l'immagine, convertendo src_points in formato np.float32
src_float = np.float32(src_points)
H = cv2.getPerspectiveTransform(src_float, dst_points)
output_img = cv2.warpPerspective(img, H, (600,800))

# Mostra l'immagine finale
cv2.namedWindow('Result', cv2.WINDOW_KEEPRATIO)
cv2.imshow('Result', output_img)
cv2.waitKey(0)

cv2.destroyAllWindows()