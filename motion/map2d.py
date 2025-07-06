import matplotlib.pyplot as plt
import numpy as np

class Map2D:
    def __init__(self):
        self.positions = []

    def add_position(self, pos):
        self.positions.append(np.array(pos))

    def plot(self, elapsed_time=None):
        positions = np.array(self.positions)
        plt.figure(figsize=(10,6))
        plt.plot(positions[:,0], positions[:,1], marker='o', linestyle='-', linewidth='2', color='red', label='Traiettoria del drone')
        plt.xlabel("X displacement", color='black')
        plt.ylabel("Y displacement", color='black')
        plt.title("FPV drone Trajectory", fontsize=16, fontweight='bold', color='black')
        plt.grid(True, linestyle='--', alpha=0.5, color='black')
        plt.axis('equal')
        stats = f"Points: {len(self.positions)}"
        if elapsed_time:
            stats += f" | Time: {elapsed_time:.2f}s"
        plt.text(0.95, 0.01, stats, ha='right', va='bottom',
                 transform=plt.gca().transAxes,
                 fontsize=9, bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))
        plt.legend()
        plt.show()
