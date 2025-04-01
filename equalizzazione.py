import cv2
import numpy as np

img = cv2.imread('01-Data/lena.png')

# Equalizzazione di un'immagine in scala di grigi
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_grey_eq = cv2.equalizeHist(img_grey)


# Equalizzazione di un'immagine a colori
eq_channels = []
channels = cv2.split(img)

for ch in channels:
    eq_channels.append(cv2.equalizeHist(ch))

img_color_eq = cv2.merge(eq_channels)


# Equalizzazione di un'immagine in HSV
# Converti l'immagine da BGR a HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_img)

# Esegui l'equalizzazione 
v_eq = cv2.equalizeHist(v)
h_eq = cv2.equalizeHist(h)
s_eq = cv2.equalizeHist(s)

equalized_hsv = cv2.merge([h, s_eq, v])

# Converti di nuovo in BGR
img_hsv_eq = cv2.cvtColor(equalized_hsv, cv2.COLOR_HSV2BGR)

cv2.imshow('Equalizzazione Grigi', img_grey_eq)
cv2.imshow('Equalizzazione Colori', img_color_eq)
cv2.imshow('Equalizzazione HSV', img_hsv_eq)
cv2.waitKey(0)