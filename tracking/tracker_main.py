import os
import numpy as np
import cv2
from sort.sort import Sort

# Config
YOLO_LABELS_PATH = "detection/runs/detect/predict5/labels"        # cartella txt YOLO predictions per frame
FRAMES_PATH = "data/frames/test_bici01"                          # cartella immagini corrispondenti
OUTPUT_PATH = "tracking/outputs/tracked"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Inizializza tracker
tracker = Sort()

# Processa tutti i frame in ordine
frames = sorted([f for f in os.listdir(FRAMES_PATH) if f.endswith(".jpg")])

for frame_file in frames:
    # Carica immagine
    img_path = os.path.join(FRAMES_PATH, frame_file)
    img = cv2.imread(img_path)

    # Carica detections YOLO + parsing perchè sono un cretino
    num_str = frame_file.replace("frame_", "").replace(".jpg", "")
    frame_idx = int(num_str)  # da '0000' -> 0
    label_file = os.path.join(YOLO_LABELS_PATH, f"bici-01_{frame_idx+1}.txt")

    print(f"[DEBUG] Frame: {frame_file}  →  Label: {label_file}")

    dets = []
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                x_center, y_center, w, h = map(float, parts[1:])
                # Converti da YOLO format a pixel
                H, W, _ = img.shape
                x1 = int((x_center - w / 2) * W)
                y1 = int((y_center - h / 2) * H)
                x2 = int((x_center + w / 2) * W)
                y2 = int((y_center + h / 2) * H)
                dets.append([x1, y1, x2, y2])
    dets = np.array(dets)

    # update tracker
    tracked_objects = tracker.update(dets)

    # Disegna risultati
    for d in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, d)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, f'ID {track_id}', (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Salva immagine
    out_path = os.path.join(OUTPUT_PATH, frame_file)
    cv2.imwrite(out_path, img)

print(f"Tracking completato! Immagini salvate in {OUTPUT_PATH}")
