import os
import numpy as np
import cv2
import csv
from sort.sort import Sort

# Config
YOLO_LABELS_PATH = "detection/runs/detect/predict-mucche-02/predict/labels"
FRAMES_PATH = "data/frames/demo-mucche02"
OUTPUT_PATH = "tracking/outputs/tracked-mucche02"
CSV_PATH = "tracking/outputs/tracking_data-mucche02.csv"

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
        if img is None:
            print(f"Immagine non trovata: {img_path}")
            continue

        # Ricava nome label corrispondente
        label_file = os.path.join(YOLO_LABELS_PATH, frame_file.replace(".jpg", ".txt"))
        print(f"[DEBUG] Frame: {frame_file} → Label: {label_file}")

        dets = []
        classes = []

        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = parts[0]
                    x_center, y_center, w, h = map(float, parts[1:])
                    H, W, _ = img.shape
                    x1 = int((x_center - w / 2) * W)
                    y1 = int((y_center - h / 2) * H)
                    x2 = int((x_center + w / 2) * W)
                    y2 = int((y_center + h / 2) * H)

                    # Scarta bbox invalide
                    if x2 > x1 and y2 > y1:
                        dets.append([x1, y1, x2, y2])
                        classes.append(cls_id)

        dets = np.array(dets)
        if len(dets) == 0:
            print(f"⚠️ Nessuna detection valida per il frame {frame_file}")

        # Aggiorna tracker
        tracked_objects = tracker.update(dets)

        # Disegna e salva CSV
        for trk in tracker.trackers:
            state = trk.get_state()
            x1, y1, x2, y2 = map(int, state)
            track_id = trk.id
            class_name = "?"    # classe yolo non definita

            if trk.hits >= tracker.min_hits or trk.time_since_update <= tracker.max_age:
                writer.writerow([frame_file, track_id, x1, y1, x2, y2, class_name])

                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(img, f'ID {track_id}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Salva frame con tracking
        out_path = os.path.join(OUTPUT_PATH, frame_file)
        cv2.imwrite(out_path, img)

print(f"\nTracking completato! Risultati CSV salvati in {CSV_PATH}")