import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Carica img
img = cv2.imread("Manipolazione Immagini/01-Data/lena.png")
print(img.shape)  # (512, 512, 3) 512x512 pixels, 3 channels (RGB)

# Sklearn KMeans expects data in 2D, so we need to reshape the image

# Immagazzino nella tupla le dimensioni originali dell'immagine height, width, channels
(h,w,c) = img.shape

# Reshape the image to a 2D array of pixels
img2D = img.reshape((h*w, c))

# Numero dei cluser = numero di colori da estrarre
model = KMeans(n_clusters=8)  
cluster_labels = model.fit_predict(img2D)

# Convert the centroids to int
rgb_colors = model.cluster_centers_.round(0).astype(int)

# Create a new image with the same shape as the original image
img_quantized = np.reshape(rgb_colors[cluster_labels], (h, w, c))
img_quantized = img_quantized.astype(np.uint8)

cv2.imshow("Original Image", img)
cv2.imshow("Quantized Image", img_quantized)
cv2.waitKey(0)