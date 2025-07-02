import cv2
import os

def extract_frames(video_path, output_folder, every_n_frames=30):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % every_n_frames == 0:
            frame_name = f"frame_{saved:04d}.jpg"
            cv2.imwrite(os.path.join(output_folder, frame_name), frame)
            saved += 1

        count += 1

    cap.release()
    print(f"[INFO] Estratti {saved} frame in {output_folder}")
