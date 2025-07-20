# TEST CV2
# import cv2
# img = cv2.imread("data/frames/train/bici-01/frame_0001.jpg")
# print(img.shape)


# TEST PYTORCH GPU MPS
# import torch
# print(torch.__version__, torch.backends.mps.is_available())


# TEST YOLO RAPIDO
from ultralytics import YOLO
model = YOLO("yolov8n.pt") 
results = model.predict("https://ultralytics.com/images/bus.jpg")  
print(results[0].boxes)
