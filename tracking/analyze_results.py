import csv
from collections import defaultdict
import matplotlib.pyplot as plt

CSV_PATH = "tracking/outputs/tracking_data.csv"

# Lettura csv + costruzione dizionario
tracks = defaultdict(list)

with open(CSV_PATH , newline='') as csvfile:
        reader =  csv.DictReader(csvfile)
        for row in reader:
                track_id = int(row['track_id'])
                frame = row['frame']
                tracks[track_id].append(frame)

# Calcolo stats
total_tracks = len(tracks) 
lenghts = [len(frames) for frames in tracks.values()]
avg_lenght = sum(lenghts) / total_tracks if total_tracks > 0 else 0

print(f"Numero totale di ID unici. {total_tracks}")
print(f"Lunghezza media dei track: {avg_lenght:.2f} frame")
for tid, frames in tracks.items():
    print(f" - ID {tid} e' in {len(frames)} frame")

# Plot di visualizzazione 
plt.figure(figsize=(10,6))
for tid, frames in tracks.items():
    x = [int(f.replace('frame_', '').replace('.jpg','')) for f in frames]
    y = [tid]*len(x)
    plt.plot(x, y, marker='o', label=f'ID {tid}')
plt.xlabel("Frame")
plt.ylabel("Track ID")
plt.title("Evoluzione dei track ID nel tempo")
plt.legend()
plt.grid(True)
plt.show()

