import cv2
import os

def extract_frames(video_path, output_folder, every_n_frames=1):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Impossibile aprire il video:", video_path)
        return

    frame_idx = 0
    saved_idx = 1  # Le label iniziano da 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n_frames == 0:
            out_filename = f"bici-03_{saved_idx:04d}.jpg"  # <-- 4 cifre con padding
            out_path = os.path.join(output_folder, out_filename)
            cv2.imwrite(out_path, frame)
            print(f"Frame {frame_idx} → {out_filename}")
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"\nEstrazione completata: {saved_idx - 1} frame salvati in '{output_folder}'")

# --- MODIFICARE QUI ---
if __name__ == "__main__":
    extract_frames(
        video_path="data/raw/bici-03.mp4",                #  Path del video
        output_folder="data/frames/demo-bici03",          #  Dove salvare i frame
        every_n_frames=1                                  #  Ogni quanti frame salvare 
    )