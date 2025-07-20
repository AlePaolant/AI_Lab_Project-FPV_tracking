import cv2
import numpy as np
import os
import time

def estimate_motion_sift(frames_path):
    frames = sorted([f for f in os.listdir(frames_path) if f.endswith(".jpg")])         # lettura delle immagini
    sift = cv2.SIFT_create()                                                            # inizializzazione oggetto SIFT
    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)                                   # ← NON USATO PIÙ
    bf = cv2.BFMatcher()                                                                # BFMatcher senza crossCheck per KNN

    trajectory = [np.array([0, 0], dtype=np.float32)]
    T_cumulative = np.eye(3, dtype=np.float32)

    start_time = time.time()
    for i in range(len(frames)-1):                                                      # processo i frame a coppie 
        print(f"\r[SIFT] Processing frame {i+1}/{len(frames)-1}", end='', flush=True)   

        img1 = cv2.imread(os.path.join(frames_path, frames[i]), cv2.IMREAD_GRAYSCALE)   # legge le immagini in scala di grigi
        img2 = cv2.imread(os.path.join(frames_path, frames[i+1]), cv2.IMREAD_GRAYSCALE)

        kp1, des1 = sift.detectAndCompute(img1, None)                                   # rileva e descrive i keyframes
        kp2, des2 = sift.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            continue

        # matches = bf.match(des1, des2)                                                # ← SOSTITUITO CON KNN
        # matches = sorted(matches, key=lambda x: x.distance)
        matches = bf.knnMatch(des1, des2, k=2)                                           # matching con KNN

        # Apply Lowe's ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 4:
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)           # estrae i punti corrispondenti
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        # Filtro basato sulla distanza euclidea
        dists = np.linalg.norm(pts1.squeeze() - pts2.squeeze(), axis=1)
        mask = (dists > 5) & (dists < 200)
        pts1 = pts1[mask]
        pts2 = pts2[mask]

        if len(pts1) < 4:
            continue

        M, _ = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)               # stima della trasformazione affine (RANSAC)
        if M is None:
            continue

        T = np.vstack([M, [0,0,1]])                                                     # Calcolo della posizione (motion accumulation)
        T_cumulative = T_cumulative @ np.linalg.inv(T)                                  # matrice 3x3, aggiorna la trasformazione e estrae la traslazione
        pos = T_cumulative[:2,2]
        trajectory.append(pos)                                                          # aggiunge la posizione corrente alla traiettoria

    elapsed_time = time.time() - start_time

    # Applica smooth finale alla traiettoria
    def smooth(trajectory, k=3):
        result = []
        for i in range(len(trajectory)):
            start = max(0, i - k)
            end = min(len(trajectory), i + k + 1)
            avg = np.mean(trajectory[start:end], axis=0)
            result.append(avg)
        return np.array(result)

    trajectory = smooth(trajectory, k=3)

    return np.array(trajectory), elapsed_time