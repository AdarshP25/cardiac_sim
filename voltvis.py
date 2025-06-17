#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")

# -------------------------------------------------------------
# Parameters
# -------------------------------------------------------------
nx, ny           = 12, 12
nt               = 10_000
snapshot_interval = 100
u_files = [f"data2/u_{t}.bin" for t in range(snapshot_interval,
                                             nt + 1, snapshot_interval)]

# -------------------------------------------------------------
# Pass 1 – find global min / max
# -------------------------------------------------------------
vmin =  np.inf
vmax = -np.inf
for f in tqdm(u_files, desc="Scanning range"):
    data = np.fromfile(f, dtype=np.float32)
    vmin = min(vmin, data.min())
    vmax = max(vmax, data.max())

print(f"Global colour range: [{vmin:.4g}, {vmax:.4g}]")

# -------------------------------------------------------------
# Pass 2 – create movie with fixed scale
# -------------------------------------------------------------
with imageio.get_writer("simulation_voltage_only.mp4", fps=10) as writer:
    for idx, f in enumerate(tqdm(u_files, desc="Rendering frames"), 1):
        u = np.fromfile(f, dtype=np.float32).reshape((ny, nx))

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(u,
                       origin="lower",
                       cmap="viridis",
                       vmin=vmin, vmax=vmax)      # <- fixed limits
        cbar = fig.colorbar(im, ax=ax, orientation="vertical",
                            fraction=0.046, pad=0.04, shrink=0.95)
        cbar.set_label("Voltage (a.u.)")

        ax.set_title(f"Time step: {idx * snapshot_interval}")
        ax.axis("off")
        plt.tight_layout()

        canvas = FigureCanvas(fig);  canvas.draw()
        buf, (w, h) = canvas.print_to_buffer()
        frame = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))[..., :3]

        writer.append_data(frame)
        plt.close(fig)

print("Done writing simulation_voltage_only.mp4")
