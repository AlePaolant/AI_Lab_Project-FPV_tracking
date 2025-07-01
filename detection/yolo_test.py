
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # uso il modello nano per leggerezza 

results = model('https://ultralytics.com/images/bus.jpg') #immagine predefinita test
results[0].show()