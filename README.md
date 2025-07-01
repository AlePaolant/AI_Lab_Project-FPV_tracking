# Object Tracking and Motion Estimation in FPV Videos using YOLOv8 + SORT and 2D Trajectory Mapping
Tracking multiplo di oggetti in video FPV tramite YOLOv8 (pre-addestrato) e SORT.

## Obiettivi
- Detect e track di oggetti in video FPV (bici, moto, persone)
- Assegnazione di ID persistenti agli oggetti in movimento (Multi-Object Tracking)
- Stima della traiettoria in 2D per ogni oggetto (Motion estimation)
- Visualizzazione finale del tracking su video e mappa 2D (interfaccia)

## Tech stack:
### YOLOv8 (pre-addestrato)
- Modello di object detection
- Distinguere classi come person, bicycle, motorcycle, ecc 
- Usato per estrarre bounding box per ogni frame

### SORT (Simple Online and Realtime Tracking)
- Algoritmo di tracking basato su:
    - associazione via IoU
    - Kalman Filter per predizione
- Mantiene gli ID degli oggetti nel tempo, frame per frame

### Motion Estimation
- Da posizione e tempo, stima di:
    - traiettoria 2D
    - velocità (approssimata, in pixel/frame o normalizzata)
    - direzione

### 2D Mapping
- Da tracking + posizione relativa in immagine, costruire una mappa in 2D
contenente proiezione semplificata del movimento nello spazio (top-down view)

### FPV drone
- Sfida: la scena cambia costantemente, velocità alte, vibrazioni, movimenti scattanti
Serve tracking robusto

## Librerie
- `ultralytics` (YOLOv8)
- `numpy`, `opencv-python`
- `matplotlib` o `plotly`
- `scikit-learn`
- `filterpy` (per Kalman Filter se SORT è scritto da zero)
- `pandas` (log delle traiettorie)
- `tqdm`, `rich` (progress bar + logging)

## Dataset e Validazione
- Video FPV personali (4k, sportivi, dinamici)
- Classi target: person, bicycle, motorcycle, horse, cow, skier, quad
- Video annotato manualmente come ground truth (per validazione delle traiettorie e ID)

## Autore
Progetto sviluppato da Alessandro Paolantonio per il corso di Laboratorio di Intelligenza Artificiale e Computer Vision - Università La Sapienza, 2025
