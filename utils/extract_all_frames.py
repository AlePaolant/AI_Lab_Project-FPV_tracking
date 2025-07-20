import os
import cv2
from video_utils import extract_frames

RAW_VIDEO_DIR = "data/raw"
OUTPUT_DIR = "data/frames"
EVERY_N_FRAMES = 5

# etichettare ogni >5 frame inizia ad essere sbagliato per il tipo di video che sto facendo,
# essendo FPV alcuni video sono molto dinamici e hanno molto motion blur,
# quindi si vuole un modello più preciso nel tracking frame-to-frame  

# Split per ciascun video
split_map = {
    # Train set
    "bici-01": "train",
    "bici-02": "train",
    "mucche-01": "train",
    "quad-mucche-02": "train",
    "trekking-01": "train",
    "sci-01": "train",
    "cavalli-01": "train",

    # Validation set
    "moto-02": "val",
    "trekking-02": "val",

    # Test set 
    "bici-03": "test",
    "moto-01": "test",
    "mucche-02": "test"
}

videos = [f for f in os.listdir(RAW_VIDEO_DIR) if f.endswith(".mp4")]

for video in videos:
    name = os.path.splitext(video)[0]
    split = split_map.get(name)

    if split is None:
        print(f"[SKIP] {video} non assegnato a nessuno split.")
        continue

    out_folder = os.path.join(OUTPUT_DIR, split)
    os.makedirs(out_folder, exist_ok=True)

    video_path = os.path.join(RAW_VIDEO_DIR, video)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"[INFO] Estrazione da {video} → Destinazione: {split.upper()} Set")

    cap = cv2.VideoCapture(video_path)
    idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % EVERY_N_FRAMES == 0:
            frame_name = f"{name}_{saved:04d}.jpg"
            out_path = os.path.join(out_folder, frame_name)
            cv2.imwrite(out_path, frame)
            saved += 1

        # Stampa progresso
        progress = int((idx / total_frames) * 20)
        bar = "#" * progress + "-" * (20 - progress)
        percent = min(int((idx / total_frames) * 100), 100)
        print(f"\r[PROCESS] {video} [{bar}] {percent}%", end="")

        idx += 1

    print(f"\r[PROCESS] {video} [{'#'*20}] 100%")
    cap.release()
    print(f"\n[SAVED] {saved} frames salvati in {out_folder}")

print("\nEstrazione frames completata con successo.")