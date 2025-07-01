from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Usa yolov8s.pt o altri se vuoi più qualità??

source = "../data/raw/bici-01.mp4"

# Inferenza
results = model(source, save=True, save_txt=True, conf=0.4)

# results è una lista, uno per ogni frame
for result in results:
    print(result.boxes.xyxy)     # bounding box [x1, y1, x2, y2]
    print(result.boxes.conf)     # confidenza
    print(result.boxes.cls)      # classe (intero)
