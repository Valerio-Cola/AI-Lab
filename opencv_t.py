import cv2
import numpy as np

# Load an image 
img = cv2.imread('01-Data/lena.png')

print(img.shape[1], img.shape[0], img.shape[2]) # width, height, channels
print(img.size) # number of pixels

# Display an image
cv2.imshow('Lena', img)
cv2.waitKey(0) # wait for a key press

cv2.destroyAllWindows() # close the window

cv2.imwrite('01-Data/lena_copy.png', img) # save an image

# Empty image using np
empty_img = np.zeros((512, 512, 3), dtype='uint8')
cv2.imshow('Empty Image', empty_img)
cv2.waitKey(0)

b,g,r = img[0, 0] # RGB values of the first pixel

img[:100, :100] = (255, 0, 0) # change the color of the top-left corner to blue
cv2.imwrite('corner.png', img) # save an image

crop = img[100:300, 100:300] # crop a region
cv2.imwrite('crop.png', crop) # save an image