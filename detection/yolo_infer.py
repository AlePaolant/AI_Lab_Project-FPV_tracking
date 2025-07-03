from ultralytics import YOLO

# CONFIG
MODEL_PATH = "detection/best.pt"        # modello fine-tunato
SOURCE = "data/raw/sci-01.mp4"         # video INPUT
IMG_SIZE = 960                          # dimensione input
CONF_THRESH = 0.4                       # confidenza minima
SAVE_DIR = "detection/runs/detect"      # directory output


def main():
    # Carica modello
    model = YOLO(MODEL_PATH)

    # Fai prediction
    results = model.predict(
        source=SOURCE,
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        save=True,
        save_txt=True,
        save_crop=False,
        project=SAVE_DIR
    )

    print(f"Inference completata! Risultati salvati in: {SAVE_DIR}")

if __name__ == "__main__":
    main()
