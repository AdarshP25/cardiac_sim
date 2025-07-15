#!/usr/bin/env python3
"""
Render voltage snapshots + deforming mesh (local-displacement format) to
an .mp4 movie.

• Works with ImageIO v2 **and** v3 (uses the v2 wrapper)
• Automatically clips the mesh sampling to the grid size
• Verifies that every .bin file has the expected length
• For small grids:
    nx, ny < 10   → label each vertex with its world (x, y) position
    10 ≤ nx, ny ≤ 15 → label numeric voltage (legacy behaviour)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")                           # head‑less backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.collections import LineCollection
import imageio.v2 as imageio                    # v2 interface works in v2 + v3
from tqdm import tqdm
import toml


folder_name = "1"
# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def make_sampled_mesh(world: np.ndarray, nsamp: int = 20) -> list:
    """
    Build a list of 2-point segments representing the (sub-sampled)
    horizontal & vertical grid lines.

    Parameters
    ----------
    world : (ny, nx, 2) array - world coordinates at this frame
    nsamp : desired number of sample points per axis (clipped to grid size)

    Returns
    -------
    list[ [ (x0,y0), (x1,y1) ], ... ]
    """
    ny, nx, _ = world.shape
    nsamp = min(nsamp, nx, ny)                  # never exceed the grid

    xs = np.linspace(0, nx - 1, nsamp, dtype=int)
    ys = np.linspace(0, ny - 1, nsamp, dtype=int)
    small = world[np.ix_(ys, xs)]               # (nsamp, nsamp, 2)

    segs = []
    segs.extend([[small[r, c], small[r, c + 1]]      # horizontals
                 for r in range(nsamp)
                 for c in range(nsamp - 1)])
    segs.extend([[small[r, c], small[r + 1, c]]      # verticals
                 for c in range(nsamp)
                 for r in range(nsamp - 1)])
    return segs


# ──────────────────────────────────────────────────────────────
# Read configuration
# ──────────────────────────────────────────────────────────────
cfg               = toml.load("config/1.toml")
nx                = cfg["simulation"]["nx"]
ny                = cfg["simulation"]["ny"]
nt                = cfg["simulation"]["nt"]
snapshot_interval = cfg["simulation"]["snapshot_interval"]
rest_len          = cfg["simulation"].get("rest_length_edge", 1.0)

u_files = [f"data/{folder_name}/u_{t}.bin" for t in range(0, nt, snapshot_interval)]
x_files = [f"data/{folder_name}/x_{t}.bin" for t in range(0, nt, snapshot_interval)]
assert len(u_files) == len(x_files), "u_ and x_ file counts differ"

# Reference grid X₀ (world space, never changes)
xv = np.arange(nx, dtype=np.float32) * rest_len
yv = np.arange(ny, dtype=np.float32) * rest_len
global_grid = np.stack(np.meshgrid(xv, yv, indexing="xy"), axis=-1)  # (ny,nx,2)

# ──────────────────────────────────────────────────────────────
# 1st pass – global colour limits
# ──────────────────────────────────────────────────────────────
vmin, vmax =  np.inf, -np.inf
for f in tqdm(u_files, desc="Scanning range"):
    d = np.fromfile(f, dtype=np.float32)
    if d.size != nx * ny:
        raise ValueError(f"{f} has wrong length ({d.size}, expected {nx*ny})")
    vmin, vmax = min(vmin, d.min()), max(vmax, d.max())
print(f"Global colour range: [{vmin:.4g}, {vmax:.4g}]")

# ──────────────────────────────────────────────────────────────
# 2nd pass – render movie
# ──────────────────────────────────────────────────────────────
with imageio.get_writer("simulation_with_mesh.mp4", fps=10) as writer:
    for step, (uf, xf) in enumerate(
            tqdm(list(zip(u_files, x_files)),
                 desc="Rendering frames", total=len(u_files))):

        # ---------- load data ------------------------------------
        u   = np.fromfile(uf, dtype=np.float32)
        disp= np.fromfile(xf, dtype=np.float32)
        if u.size != nx * ny:
            raise ValueError(f"{uf} has wrong length")
        if disp.size != nx * ny * 2:
            raise ValueError(f"{xf} has wrong length")

        u    =  u.reshape(ny, nx)
        disp = disp.reshape(ny, nx, 2)
        world = global_grid + disp              # X = X₀ + u

        # ---------- mesh segments --------------------------------
        mesh_lines = make_sampled_mesh(world, nsamp=20)

        # ---------- plotting ------------------------------------
        fig, ax = plt.subplots(figsize=(6, 6))

        im = ax.imshow(u,
                       origin="lower",
                       extent=(-0.5, nx - 0.5, -0.5, ny - 0.5),
                       cmap="viridis",
                       vmin=vmin, vmax=vmax)

        lc = LineCollection(mesh_lines, colors="white",
                            linewidths=0.7, alpha=0.9)
        ax.add_collection(lc)

        # ———— overlays ————
        if nx < 10 and ny < 10:
            # label vertex *world* coordinates
            for j in range(ny):
                for i in range(nx):
                    x, y = world[j, i]
                    ax.text(i, j,
                            f"({x:.2f}, {y:.2f})",
                            color="white", fontsize=6,
                            ha="center", va="center")
        elif nx <= 15 and ny <= 15:
            # label voltage values
            for j in range(ny):
                for i in range(nx):
                    ax.text(i, j,
                            f"{u[j, i]:.2f}",
                            color="white", fontsize=6,
                            ha="center", va="center")

        cbar = fig.colorbar(im, ax=ax, orientation="vertical",
                            fraction=0.046, pad=0.04, shrink=0.95)
        cbar.set_label("Voltage (a.u.)")

        ax.set_title(f"Time step: {step * snapshot_interval}")
        ax.axis("off")
        plt.tight_layout()

        # ---------- write frame ----------------------------------
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf, (w, h) = canvas.print_to_buffer()
        frame = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))[..., :3]
        writer.append_data(frame)
        plt.close(fig)

print("Done writing simulation_with_mesh.mp4")
