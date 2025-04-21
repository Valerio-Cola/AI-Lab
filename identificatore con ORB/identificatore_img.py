import cv2
import os

# Path alle immagini
db_path = "identificatore con ORB/database/"
query_path = "identificatore con ORB/query/"

# Nomi delle immagini (database)
cover_filenames = []
for filename in os.listdir(db_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        cover_filenames.append(filename)

# Inizializza ORB e lista per i descrittori
orb = cv2.ORB_create(nfeatures=1000)
descriptors_list = []

# Estrazione descrittori dal database
for filename in cover_filenames:
    # Carica la copertina incollando database/ e il nome della copertina
    # (es. database/necronomicon.jpg)
    img_path = os.path.join(db_path, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Trova i keypoints e calcola i descrittori inserendoli nella lista 
    kp, des = orb.detectAndCompute(img, None)
    descriptors_list.append(des)

# Carica immagine da classificare
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

for filename in os.listdir(query_path):

    query_img = cv2.imread(os.path.join(query_path, filename), cv2.IMREAD_GRAYSCALE)

    # Estrae i keypoints e i descrittori dell'immagine da classificare
    kp_query, des_query = orb.detectAndCompute(query_img, None)

    # Crea lista dei match
    matches_list = []

    # Per ogni gruppo di descrittori nella lista
    for des_db in descriptors_list:
        # Se il descrittore del database e quello della query non sono vuoti
        if des_db is not None and des_query is not None:
            # Trova i match tra i due descrittori, li ordina e aggiunge alla lista dei match
            matches = bf.match(des_query, des_db)
            matches = sorted(matches, key=lambda x: x.distance)
            matches_list.append(matches)
        else:
            matches_list.append([])

    # Trova il match con più good matches in modo da limitare i falsi positivi
    threshold = 30  # soglia per robustezza (step 7)

    best_index = -1 # indice del match migliore
    max_good_matches = 0 # variabile per il numero di good matches trovati 


    # Per ogni match trovato nel database conta quanti sono i good matches
    for i, matches in enumerate(matches_list):

        good = []
        for match in matches:
            if match.distance < 60:
                good.append(match)

        print(f"[INFO] Oggetto {i+1}: {len(good)} good matches")

        # Trova il match con più good matches
        if len(good) > max_good_matches:
            max_good_matches = len(good)
            best_filename = cover_filenames[i]
            best_index = i

    # Se nel migliore match ci sono più good matches della soglia, stampa il risultato
    skip = False
    if max_good_matches > threshold:
        print(f"✅ L'immagine è stata classificata come: {best_filename} "
              f"con {max_good_matches} good matches\n")
    else:
        print(f"⚠️ Nessuna classificazione affidabile. Solo {max_good_matches} good matches.\n")
        skip = True


    # === Visualizzazione opzionale dei match ===
    if max_good_matches > threshold and not skip:
        # Carica l'immagine del database corrispondente e ricarica solo i keypoints
        db_img = cv2.imread(os.path.join(db_path, best_filename), cv2.IMREAD_GRAYSCALE)
        kp_db, _ = orb.detectAndCompute(db_img, None)
        
        # Ricalcola i good match
        good_matches = [m for m in matches_list[best_index] if m.distance < 60]

        # Li disegna mettendo in cofronto l'immagine di query e del database
        match_img = cv2.drawMatches(query_img, kp_query, db_img, kp_db, good_matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        cv2.namedWindow("Matching", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Matching", match_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def objClassification(frame,descriptors_list):
    orb = cv2.ORB_create(nfeatures=1000)
    kp, des = orb.detectAndCompute(frame,None)

    matcher = cv2.BFMatcher()
    best_matches = []

    for des_db in descriptors_list:
        matches = matcher.knnMatch(des,des_db,k=2)
        good = []

        for m,n in matches:
            if m.distance < n.distance * 0.8:
                good.append([m])

    best_matches.append(len(good))
    
    best_index = -1
    if len(best_matches) > 0:
        max_val = max(best_matches)
        if max_val > 10:
            best_index = best_matches.index(max_val)
    

    return cover_filenames[best_index]
    
# Mediante webcam, ma anche ip, link, folder
webcam = cv2.VideoCapture(0)


while True:
    # Lettura del singolo frame
    succes, frame = webcam.read()

    obj = objClassification(frame, descriptors_list)

    if obj != None:
        cv2.putText(frame, obj, (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

    cv2.imshow('Frame', frame)
    k = cv2.waitKey(30)
    if k == ord('q'):
        break



