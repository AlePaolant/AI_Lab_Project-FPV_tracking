import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time

FRAMES_PATH = "data/frames/test_bici01"  # metti la tua cartella
frames = sorted([f for f in os.listdir(FRAMES_PATH) if f.endswith(".jpg")])

# Inizializza SIFT e BFMatcher (normalizzazione L2)
sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Posizione iniziale (origine)
trajectory = [np.array([0, 0], dtype=np.float32)]

# Trasformazione cumulata (2x3 affine)
T_cumulative = np.eye(3, dtype=np.float32)

start_time = time.time()

for i in range(len(frames)-1):
    print(f"[STATUS] Processing frame {i+1}/{len(frames)-1}")
    img1 = cv2.imread(os.path.join(FRAMES_PATH, frames[i]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(FRAMES_PATH, frames[i+1]), cv2.IMREAD_GRAYSCALE)

    # Trova keypoints SIFT e calcola descrizioni
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Skip se non trova niente
    if des1 is None or des2 is None:
        continue

    # Match descriptors - brute force
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) < 4:
        continue

    # Estrai punti corrispondenti
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    # Stima matrice affine (puoi usare anche cv2.findHomography con RANSAC)
    M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)
    if M is None:
        continue

    # Trasforma punto (0,0) tramite la matrice cumulata
    T = np.vstack([M, [0,0,1]])  # passa da 2x3 a 3x3
    T_cumulative = T_cumulative @ np.linalg.inv(T)

    # Salva la posizione corrente (il punto origine trasformato)
    pos = T_cumulative[:2,2]
    trajectory.append(pos)

end_time = time.time()
elapsed_time = end_time - start_time

# Plot trajectory
trajectory = np.array(trajectory)
plt.figure(figsize=(8,6))
plt.plot(trajectory[:,0], trajectory[:,1], marker='o')
plt.title("Stima traiettoria del drone (frame-to-frame con SIFT)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.text(0.95, 0.01, f"Time: {elapsed_time:.2f}s",
         ha='right', va='bottom', transform=plt.gca().transAxes,
         fontsize=9, bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

plt.show()
