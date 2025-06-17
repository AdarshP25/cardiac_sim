#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.collections import LineCollection
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")

# -------------------------------------------------------------
# Parameters
# -------------------------------------------------------------
nx, ny            = 12, 12
nt                = 10_000
snapshot_interval = 100

u_files = [f"data2/u_{t}.bin" for t in range(snapshot_interval,
                                             nt + 1, snapshot_interval)]
x_files = [f"data2/x_{t}.bin" for t in range(snapshot_interval,
                                             nt + 1, snapshot_interval)]
assert len(u_files) == len(x_files), "u_ and x_ file counts differ"

# -------------------------------------------------------------
# Build helper to convert a (ny,nx,2) array → list[[(x0,y0),(x1,y1)], …]
# -------------------------------------------------------------
def make_mesh_segments(coords: np.ndarray) -> list:
    """
    coords  shape (ny , nx , 2) – each entry is (x,y) of a vertex.
    Returns list of 2‑point segments for horizontal & vertical edges.
    """
    segs = []
    # horizontal edges
    segs.extend(
        [[coords[r, c], coords[r, c + 1]]
         for r in range(coords.shape[0])
         for c in range(coords.shape[1] - 1)]
    )
    # vertical edges
    segs.extend(
        [[coords[r, c], coords[r + 1, c]]
         for c in range(coords.shape[1])
         for r in range(coords.shape[0] - 1)]
    )
    return segs

# -------------------------------------------------------------
# Pass 1 – global min / max
# -------------------------------------------------------------
vmin =  np.inf
vmax = -np.inf
for f in tqdm(u_files, desc="Scanning range"):
    data = np.fromfile(f, dtype=np.float32)
    vmin = min(vmin, data.min())
    vmax = max(vmax, data.max())
print(f"Global colour range: [{vmin:.4g}, {vmax:.4g}]")

# -------------------------------------------------------------
# Pass 2 – render movie with fixed scale + deforming grid
# -------------------------------------------------------------
with imageio.get_writer("simulation_with_mesh.mp4", fps=10) as writer:
    for step, (uf, xf) in enumerate(tqdm(zip(u_files, x_files),
                                         desc="Rendering frames",
                                         total=len(u_files)), 1):
        # --- load data ------------------------------------------------
        u   = np.fromfile(uf, dtype=np.float32).reshape((ny, nx))
        pos = np.fromfile(xf, dtype=np.float32).reshape((ny, nx, 2))

        # --- build mesh segments (every vertex) ----------------------
        mesh_lines = make_mesh_segments(pos)

        # --- plotting ------------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(u, origin="lower", cmap="viridis",
                       vmin=vmin, vmax=vmax)
        lc = LineCollection(mesh_lines, colors="white",
                            linewidths=0.7, alpha=0.9)
        ax.add_collection(lc)

        cbar = fig.colorbar(im, ax=ax, orientation="vertical",
                            fraction=0.046, pad=0.04, shrink=0.95)
        cbar.set_label("Voltage (a.u.)")

        ax.set_title(f"Time step: {step * snapshot_interval}")
        ax.axis("off")
        plt.tight_layout()

        # --- write frame ---------------------------------------------
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf, (w, h) = canvas.print_to_buffer()
        frame = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))[..., :3]
        writer.append_data(frame)
        plt.close(fig)

print("Done writing simulation_with_mesh.mp4")
