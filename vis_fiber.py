#!/usr/bin/env python3
"""
show_fibers_rgb.py    —  visualize an axial fiber map as a pure color image
Usage:
    python show_fibers_rgb.py  fibers.bin  nx  ny
    (defaults: fibers.bin 202 202)

The file must contain (ny-1)*(nx-1) float32 angles in row-major order.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------- command-line or defaults -----------------------------------------
bin_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fibers.bin")
nx       = int(sys.argv[2])  if len(sys.argv) > 2 else 256
ny       = int(sys.argv[3])  if len(sys.argv) > 3 else 256
shape    = (ny - 1, nx - 1)

# -------- load map ----------------------------------------------------------
theta = np.fromfile(bin_file, dtype=np.float32)
if theta.size != shape[0] * shape[1]:
    raise ValueError(f"File size does not match grid {shape}")

theta = theta.reshape(shape)

# -------- convert axial angle → RGB via HSV ---------------------------------
#   axial ⇒ double the phase so θ and θ+π map to the same hue
phase      = (2.0 * theta) % (2*np.pi)      # [0, 2π)
hue        = phase / (2*np.pi)              # [0,1) for HSV
sat        = np.ones_like(hue)              # full saturation
val        = np.ones_like(hue)              # full value (brightness)

# HSV → RGB (vectorised)
import matplotlib.colors as mcolors
rgb = mcolors.hsv_to_rgb(np.stack((hue, sat, val), axis=-1))

# -------- display -----------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.imshow(rgb, origin="upper", interpolation="nearest")
plt.axis("off")
plt.title("Fiber orientation (cyclic hue)")

# optional inset color wheel legend
ax = plt.gca()
wheel = plt.axes([0.78, 0.78, 0.18, 0.18], polar=True)
t     = np.linspace(0, 2*np.pi, 256)
wheel.bar(t, np.ones_like(t), width=t[1]-t[0],
          color=plt.cm.hsv(t/(2*np.pi)), edgecolor='none')
wheel.set_yticklabels([])
wheel.set_xticklabels([])
wheel.set_title("axis\n0°–π", va='bottom', fontsize=8)

plt.tight_layout()
plt.show()
