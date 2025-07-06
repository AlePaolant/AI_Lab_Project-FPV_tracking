import cv2
import numpy as np
import os
import time

def estimate_motion_orb(frames_path):
    frames = sorted([f for f in os.listdir(frames_path) if f.endswith(".jpg")])
    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    trajectory = [np.array([0, 0], dtype=np.float32)]
    T_cumulative = np.eye(3, dtype=np.float32)

    start_time = time.time()
    for i in range(len(frames)-1):
        print(f"\r[ORB] Processing frame {i+1}/{len(frames)-1}", end='', flush=True)

        img1 = cv2.imread(os.path.join(frames_path, frames[i]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(os.path.join(frames_path, frames[i+1]), cv2.IMREAD_GRAYSCALE)

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            continue

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < 4:
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

        M, _ = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)
        if M is None:
            continue

        T = np.vstack([M, [0,0,1]])
        T_cumulative = T_cumulative @ np.linalg.inv(T)
        pos = T_cumulative[:2,2]
        trajectory.append(pos)

    elapsed_time = time.time() - start_time
    return np.array(trajectory), elapsed_time
