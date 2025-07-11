import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_hex

import numpy as np
import csv

class Map2D:
    def __init__(self):
        self.positions = []
        self.orientations = []
        self.objects_per_frame = {}
        self.all_ids = set()

    def add_position(self, pos):
        self.positions.append(np.array(pos))
        if len(self.positions) >= 2:
            dx = self.positions[-1][0] - self.positions[-2][0]
            dy = self.positions[-1][1] - self.positions[-2][1]
            angle = np.arctan2(dy, dx) + np.pi/2  # aggiusta per gli orbit FPV
            self.orientations.append(angle)
        elif len(self.positions) == 1:
            self.orientations.append(0.0)

    def load_tracking_data(self, csv_path, frame_width=1280):
        self.objects_per_frame = {}
        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                try:
                    frame_str = row[0]
                    frame_idx = int(frame_str.replace("frame_","").replace(".jpg",""))
                    obj_id = int(row[1])
                    x1, y1, x2, y2 = map(int, row[2:6])

                    x_center = (x1 + x2) / 2
                    bbox_height = y2 - y1

                    self.all_ids.add(obj_id)
                    if frame_idx not in self.objects_per_frame:
                        self.objects_per_frame[frame_idx] = []
                    self.objects_per_frame[frame_idx].append((x_center, bbox_height, obj_id))
                except:
                    continue  # skip righe problematiche

    def plot(self, elapsed_time=None, W=1280):
        positions = np.array(self.positions)
        plt.figure(figsize=(12,8))
        ax = plt.gca()
        ax.set_facecolor('#121212')
        plt.plot(positions[:,0], positions[:,1],
                 color='red', linestyle='-', linewidth=2, marker='o',
                 label='Drone trajectory')
        
        palette = [to_hex(c) for c in cm.nipy_spectral(np.linspace(0, 1, 20))]
        
        forward_radius = 50.0
        side_offset = 150.0  # quanto traslare TUTTA la bolla lateralmente (a destra del drone)

        for idx in range(len(self.positions)):
            base_pos = positions[idx]
            angle_cam = self.orientations[min(idx, len(self.orientations)-1)]
            objs = self.objects_per_frame.get(idx, [])

            # offset frontale
            forward_x = forward_radius * np.cos(angle_cam - np.pi/2)
            forward_y = forward_radius * np.sin(angle_cam - np.pi/2)

            # offset laterale
            lateral_x = -side_offset * np.cos(angle_cam)
            lateral_y = -side_offset * np.sin(angle_cam)

            for (x_center, bbox_height, obj_id) in objs:
                final_x = base_pos[0] + forward_x + lateral_x
                final_y = base_pos[1] + forward_y + lateral_y

                color = palette[obj_id % len(palette)]
                plt.plot(final_x, final_y, marker='o', markersize=4, color=color)



        handles = []
        labels = []
        for obj_id in sorted(self.all_ids):
            color = palette[obj_id % len(palette)]
            handles.append(plt.Line2D([], [], color=color, marker='o', linestyle='None'))
            labels.append(f'ID {obj_id}')
        plt.legend(handles, labels, frameon=False, labelcolor='white')

        plt.xlabel("X", fontsize=12, color='white')
        plt.ylabel("Y", fontsize=12, color='white')
        plt.title("2D Map with FPV Drone Trajectory and Projected Objects", fontsize=16, color='white')
        plt.grid(True, linestyle='--', alpha=0.5, color='white')
        plt.axis('equal')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')

        stats = f"Frames: {len(self.positions)}"
        if elapsed_time:
            stats += f" | Time: {elapsed_time:.2f}s"
        plt.text(0.98, 0.02, stats, ha='right', va='bottom',
                 transform=ax.transAxes, fontsize=10,
                 bbox=dict(facecolor='black', alpha=0.8, edgecolor='white'), color='white')

        plt.legend(frameon=False, labelcolor='white')
        plt.show()
