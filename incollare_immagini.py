import cv2
import numpy as np  

# Query Image
img1 = cv2.imread('01-Data/right.png') 
# Train Image
img2 = cv2.imread('01-Data/left.png')  

img1 = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
img2 = cv2.resize(img2, (0, 0), fx=0.5, fy=0.5)

orb = cv2.ORB_create()

kpts1, des1 = orb.detectAndCompute(img1, None)
kpts2, des2 = orb.detectAndCompute(img2, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = matcher.knnMatch(des1, des2, k=2)

# Ratio Test
# Keep only good matches using the Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.5 * n.distance:
        good_matches.append(m)

# Draw the keypoints
img1_kpts = cv2.drawKeypoints(img1, kpts1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_kpts = cv2.drawKeypoints(img2, kpts2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Keypoints Image 1', img1_kpts)
cv2.imshow('Keypoints Image 2', img2_kpts)
cv2.waitKey(0)

# Draw the matches
img_matches = cv2.drawMatches(img1, kpts1, img2, kpts2, good_matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('Matches', img_matches)
cv2.waitKey(0)


# Verifica che ci siano almeno 4 match per la matrice di omografia
# QueryIdx e TrainIdx sono gli indici dei punti chiave rispettivamente di img1 e img2
if len(good_matches) >= 4:

    # In questo modo accedo ai punti chiave delle immagini
    # Src_points saranno i punti in cui verranno incollati i punti di dst_points
    src_points = np.float32([kpts1[m.queryIdx].pt for m in good_matches])
    dst_points = np.float32([kpts2[m.trainIdx].pt for m in good_matches])

    # Calcolo la matrice di omografia
    M, mask = cv2.findHomography(src_points, dst_points)

    # Trasformo l'immagine 1 in modo che ci sia spazio per incollare l'immagine 2
    dst = cv2.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1] - int(M[0,2]), img1.shape[0]))
    
    cv2.namedWindow('Warped Image', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Warped Image', dst)
    cv2.waitKey(0)

    dst[0:img2.shape[0], 0:img2.shape[1]] = img2.copy()
    cv2.namedWindow('Final Image', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Final Image', dst)
    cv2.waitKey(0)