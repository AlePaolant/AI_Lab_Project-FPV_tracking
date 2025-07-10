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
mappa.load_tracking_data("tracking/outputs/tracking_data.csv")

mappa.plot(elapsed_time)