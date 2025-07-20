import os
import numpy as np
import cv2
import csv
from deep_sort_realtime.deepsort_tracker import DeepSort

# Config
YOLO_LABELS_PATH = "detection/runs/detect/predict5/labels"
FRAMES_PATH = "data/frames/test_bici01"
OUTPUT_PATH = "tracking/outputs/tracked_deepsort"
CSV_PATH = "tracking/outputs/tracking_data_deepsort.csv"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Inizializza DeepSORT tracker
tracker = DeepSort(max_age=15, n_init=2, nms_max_overlap=1.0)

# Apri CSV
with open(CSV_PATH, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "class"])

    # Carica frame
    frames = sorted([f for f in os.listdir(FRAMES_PATH) if f.endswith(".jpg")])
    for frame_file in frames:
        img_path = os.path.join(FRAMES_PATH, frame_file)
        img = cv2.imread(img_path)

        # Ricava YOLO txt
        num_str = frame_file.replace("frame_", "").replace(".jpg", "")
        frame_idx = int(num_str)
        label_file = os.path.join(YOLO_LABELS_PATH, f"bici-01_{frame_idx+1}.txt")
        print(f"[DEBUG] Frame: {frame_file} → Label: {label_file}")

        detections = []
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

                    detections.append(([x1, y1, x2 - x1, y2 - y1], 0.99, cls_id))  # (tlwh, conf, class)
                    classes.append(cls_id)

        # Tracking
        tracks = tracker.update_tracks(detections, frame=img)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, w, h = track.to_ltrb()
            x1, y1, x2, y2 = int(l), int(t), int(w), int(h)
            class_name = track.get_det_class() or "?"

            writer.writerow([frame_file, track_id, x1, y1, x2, y2, class_name])

            # Disegna
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.putText(img, f'DS {track_id}', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        out_path = os.path.join(OUTPUT_PATH, frame_file)
        cv2.imwrite(out_path, img)

print(f"DeepSORT tracking completato! CSV salvato in {CSV_PATH}")
