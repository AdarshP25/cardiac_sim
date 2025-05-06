#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm  # progress bar

# Force Agg backend
import matplotlib
matplotlib.use('Agg')

# Parameters
nx, ny = 1002, 1002
nt = 100000
snapshot_interval = 1000

# Precompute a 100×100 sampling grid of indices
sample_i = np.linspace(0, nx - 1, 20, dtype=int)
sample_j = np.linspace(0, ny - 1, 20, dtype=int)

def make_mesh_segments(coords):
    """
    Given coords shape (100,100,2), return list of line segments
    for both horizontal and vertical mesh lines.
    """
    segments = []
    # horizontal
    for r in range(coords.shape[0]):
        row = coords[r]
        for c in range(coords.shape[1] - 1):
            segments.append([row[c], row[c+1]])
    # vertical
    for c in range(coords.shape[1]):
        col = coords[:, c]
        for r in range(coords.shape[0] - 1):
            segments.append([col[r], col[r+1]])
    return segments

# Gather file paths
u_files = [f"data2/u_{t}.bin" for t in range(snapshot_interval, nt+1, snapshot_interval)]
x_files = [f"data2/x_{t}.bin" for t in range(snapshot_interval, nt+1, snapshot_interval)]

# Open the video writer
with imageio.get_writer("simulation_with_mesh.mp4", fps=10) as writer:
    for idx in tqdm(range(len(u_files)), desc="Rendering frames"):
        t = (idx + 1) * snapshot_interval
        u_path = u_files[idx]
        x_path = x_files[idx]

        # Load data
        u = np.fromfile(u_path, dtype=np.float32).reshape((ny, nx))
        pos = np.fromfile(x_path, dtype=np.float32).reshape((ny, nx, 2))
        coords = pos[sample_j[:, None], sample_i[None, :]]  # (100,100,2)

        # Plot frame
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(u, origin='lower', cmap='viridis')
        mesh_lines = make_mesh_segments(coords)
        lc = LineCollection(mesh_lines, colors='white', linewidths=0.5)
        ax.add_collection(lc)
        ax.set_title(f"Time step: {t}")
        ax.axis('off')

        # Capture RGBA buffer from Agg canvas
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf, (w, h) = canvas.print_to_buffer()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        rgb_frame = arr[..., :3]   # drop alpha channel

        writer.append_data(rgb_frame)
        plt.close(fig)

print("Done writing simulation_with_mesh.mp4")
