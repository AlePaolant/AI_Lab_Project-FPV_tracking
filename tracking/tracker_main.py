import os
import numpy as np
import cv2
import csv
from sort.sort import Sort

# Config
YOLO_LABELS_PATH = "detection/runs/detect/predict5/labels"
FRAMES_PATH = "data/frames/test_bici01"
OUTPUT_PATH = "tracking/outputs/tracked"
CSV_PATH = "tracking/outputs/tracking_data.csv"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Inizializza tracker
tracker = Sort()

# Apri CSV in scrittura
with open(CSV_PATH, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "class"])

    # Processa tutti i frame
    frames = sorted([f for f in os.listdir(FRAMES_PATH) if f.endswith(".jpg")])
    for frame_file in frames:
        img_path = os.path.join(FRAMES_PATH, frame_file)
        img = cv2.imread(img_path)

        # Ricava nome file YOLO
        num_str = frame_file.replace("frame_", "").replace(".jpg", "")
        frame_idx = int(num_str)
        label_file = os.path.join(YOLO_LABELS_PATH, f"bici-01_{frame_idx+1}.txt")
        print(f"[DEBUG] Frame: {frame_file} → Label: {label_file}")

        dets = []
        classes = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    cls_id = parts[0]
                    x_center, y_center, w, h = map(float, parts[1:])
                    H, W, _ = img.shape
                    x1 = int((x_center - w / 2) * W)
                    y1 = int((y_center - h / 2) * H)
                    x2 = int((x_center + w / 2) * W)
                    y2 = int((y_center + h / 2) * H)
                    dets.append([x1, y1, x2, y2])
                    classes.append(cls_id)

        dets = np.array(dets)

        # Aggiorna tracker
        tracked_objects = tracker.update(dets)

        # Disegna e salva CSV
        for i, d in enumerate(tracked_objects):
            x1, y1, x2, y2, track_id = map(int, d)
            class_name = classes[i] if i < len(classes) else "?"
            writer.writerow([frame_file, track_id, x1, y1, x2, y2, class_name])

            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, f'ID {track_id}', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Salva frame con tracking
        out_path = os.path.join(OUTPUT_PATH, frame_file)
        cv2.imwrite(out_path, img)

print(f"Tracking completato! Risultati CSV salvati in {CSV_PATH}")
