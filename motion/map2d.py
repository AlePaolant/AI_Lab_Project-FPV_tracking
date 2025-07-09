import matplotlib.pyplot as plt
import numpy as np

class Map2D:
    def __init__(self):
        self.positions = []
        self.objects = []

    def add_position(self, pos):
        self.positions.append(np.array(pos))
        self.objects.append([])  # crea lista vuota per oggetti in quel frame

    def add_objects_at_frame(self, frame_idx, objects):
        self.objects[frame_idx] = objects

    def plot(self, elapsed_time=None):
        positions = np.array(self.positions)
        plt.figure(figsize=(12,8))
        ax = plt.gca()
        ax.set_facecolor('#121212')
        plt.plot(positions[:,0], positions[:,1],
                 color='red', linestyle='-', linewidth=2, marker='o',
                 label='Drone trajectory')

        # Plot oggetti
        import random
        colors = {}
        for idx, objs in enumerate(self.objects):
            base_pos = positions[idx]
            for obj in objs:
                xoff, yoff, obj_id = obj
                if obj_id not in colors:
                    colors[obj_id] = [random.random(), random.random(), random.random()]
                color = colors[obj_id]
                plt.plot(base_pos[0]+xoff, base_pos[1]+yoff, marker='o', markersize=6, color=color)

        plt.xlabel("X", fontsize=12, color='white')
        plt.ylabel("Y", fontsize=12, color='white')
        plt.title("Drone Map with Tracked Objects", fontsize=16, color='white')
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
