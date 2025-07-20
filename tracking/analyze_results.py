import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

CSV_SORT = "tracking/outputs/tracking_data-mucche02.csv"
CSV_DEEPSORT = "tracking/outputs/tracking_data_deepsort_mucche02.csv"

def process_csv(csv_path):
    tracks = defaultdict(list)
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            track_id = int(row['track_id'])
            # Estrae numero frame da 'bici-03_0001.jpg'
            try:
                frame_str = row['frame'].replace('.jpg', '')
                frame_num = ''.join(filter(str.isdigit, frame_str))
                frame = int(frame_num)
            except Exception as e:
                print(f"Errore parsing frame: {row['frame']} → {e}")
                continue
            tracks[track_id].append(frame)
    return tracks

def compute_stats(tracks, name=""):
    total_tracks = len(tracks)
    lengths = [len(frames) for frames in tracks.values()]
    avg_length = np.mean(lengths) if total_tracks > 0 else 0
    max_length = np.max(lengths) if total_tracks > 0 else 0
    min_length = np.min(lengths) if total_tracks > 0 else 0
    std_dev = np.std(lengths) if total_tracks > 0 else 0

    print(f"\nStatistiche per {name}")
    print(f" - Numero totale di ID unici: {total_tracks}")
    print(f" - Lunghezza media dei track: {avg_length:.2f} frame")
    print(f" - Lunghezza massima dei track: {max_length} frame")
    print(f" - Lunghezza minima dei track: {min_length} frame")
    print(f" - Deviazione standard: {std_dev:.2f} frame")

    return lengths

# Processa entrambi i CSV
sort_tracks = process_csv(CSV_SORT)
deepsort_tracks = process_csv(CSV_DEEPSORT)

# Calcola statistiche
sort_lengths = compute_stats(sort_tracks, "SORT")
deepsort_lengths = compute_stats(deepsort_tracks, "DeepSORT")

# Visualizzazione: tracking nel tempo
fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# --- SORT
for tid, frames in sort_tracks.items():
    axs[0].plot(frames, [tid] * len(frames), marker='o', linestyle='-', markersize=3)
axs[0].set_title("SORT - Evoluzione dei Track ID")
axs[0].set_xlabel("Frame")
axs[0].set_ylabel("Track ID")
axs[0].grid(True)

# --- DeepSORT
for tid, frames in deepsort_tracks.items():
    axs[1].plot(frames, [tid] * len(frames), marker='o', linestyle='-', markersize=3)
axs[1].set_title("DeepSORT - Evoluzione dei Track ID")
axs[1].set_xlabel("Frame")
axs[1].grid(True)

plt.tight_layout()
plt.show()