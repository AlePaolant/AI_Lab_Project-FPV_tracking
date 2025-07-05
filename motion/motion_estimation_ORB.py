import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time


FRAMES_PATH = "data/frames/test_bici01"
frames = sorted([f for f in os.listdir(FRAMES_PATH) if f.endswith(".jpg")])

# ORB detector + Hamming distance matcher
orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

trajectory = [np.array([0, 0], dtype=np.float32)]
T_cumulative = np.eye(3, dtype=np.float32)

start_time = time.time()

total = len(frames) - 1

for i in range(total):
    print(f"[INFO] Processing frame {i+1}/{total} (ORB)")
    img1 = cv2.imread(os.path.join(FRAMES_PATH, frames[i]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(FRAMES_PATH, frames[i + 1]), cv2.IMREAD_GRAYSCALE)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        continue

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 4:
        continue

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)
    if M is None:
        continue

    T = np.vstack([M, [0, 0, 1]])
    T_cumulative = T_cumulative @ np.linalg.inv(T)
    pos = T_cumulative[:2, 2]
    trajectory.append(pos)

    # direzione "in avanti" (asse x trasformato)
    direction_vector = T_cumulative[:2, 0]

end_time = time.time()
elapsed_time = end_time - start_time

trajectory = np.array(trajectory)

plt.figure(figsize=(10, 6))
plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', linestyle='-', color='orange')
plt.title("Estimated 2D Drone Trajectory via ORB motion estimation")
plt.xlabel("X displacement")
plt.ylabel("Y displacement")
plt.grid(True)
plt.legend()

plt.text(0.95, 0.01, f"Time: {elapsed_time:.2f}s",
         ha='right', va='bottom', transform=plt.gca().transAxes,
         fontsize=9, bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

plt.show()
