import os

# === CONFIGURAZIONE ===
LABELS_DIR = "detection/runs/detect/predict-mucche-02/predict/labels"
FILENAME_PREFIX = "mucche-02"       # Cambia se il prefisso è diverso
PADDING = 4                       # bici-03_0001.txt → padding a 4 cifre

# === ESECUZIONE ===
def rename_labels():
    if not os.path.exists(LABELS_DIR):
        print(f"Cartella labels non trovata: {LABELS_DIR}")
        return

    renamed = 0
    for filename in os.listdir(LABELS_DIR):
        if not filename.endswith(".txt"):
            continue

        name, _ = os.path.splitext(filename)   # esempio: bici-03_1
        parts = name.split("_")

        if len(parts) == 2 and parts[1].isdigit():
            index = int(parts[1])
            new_name = f"{FILENAME_PREFIX}_{index:0{PADDING}d}.txt"

            old_path = os.path.join(LABELS_DIR, filename)
            new_path = os.path.join(LABELS_DIR, new_name)

            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"{filename} → {new_name}")
                renamed += 1

    print(f"\nCompletato: {renamed} file rinominati.")

if __name__ == "__main__":
    rename_labels()