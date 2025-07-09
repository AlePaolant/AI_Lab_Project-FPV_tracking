from motion_estimation_SIFT import estimate_motion_sift
from motion_estimation_ORB import estimate_motion_orb
from map2d import  Map2D
import csv


FRAMES_PATH = "data/frames/test_bici01"

print("Seleziona il metodo:")
print("0 - Annulla")
print("1 - SIFT")
print("2 - ORB")
sel = input("")

if sel == "0":
    exit()
elif sel == "1":
    print("Avvio motion estimation SIFT...")
    trajectory, elapsed_time = estimate_motion_sift(FRAMES_PATH)
    print("\nMotion estimation SIFT completata!")
elif sel == "2":
    print("Avvio motion estimation ORB...")
    trajectory, elapsed_time = estimate_motion_orb(FRAMES_PATH)
    print("\nMotion estimation ORB completata!")
else:
    print ("Invalido - annullo.")
    exit()

# crea mappa
print("Creo mappa...")
mappa = Map2D()
for pos in trajectory:
    mappa.add_position(pos)

# aggiungi oggetti tracciati
print("Carico tracking data...")
tracking_data = {}
with open("tracking/outputs/tracking_data.csv") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        try:
            frame_str = row[0]
            frame = int(frame_str.replace("frame_","").replace(".jpg",""))
            obj_id = int(row[1])
            x1, y1, x2, y2 = map(int, row[2:6])
            class_id = int(row[6])
            
            W, H = 1280, 720
            x_center = (x1 + x2) / 2
            bbox_height = y2 - y1

            H_ref = 200
            k_scale = 5.0
            d_rel = k_scale * (H_ref / max(bbox_height, 1))

            x_offset = (x_center - W/2) / (W/2) * d_rel
            y_offset = -d_rel

            tracking_data.setdefault(frame, []).append((x_offset, y_offset, obj_id))
        
        except Exception as e:
            print(f"[WARNING] Problema parsing riga {row}: {e}")


for idx in range(len(trajectory)):
    objs = tracking_data.get(idx, [])
    mappa.add_objects_at_frame(idx, objs)



mappa.plot(elapsed_time)