import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
nx = 1002
ny = 1002
nt = 1000001
snapshot_interval = 1000
num_snapshots = nt // snapshot_interval

# Load snapshots
frames = np.zeros((num_snapshots, nx, ny))
for i in range(num_snapshots):
    t = (i + 1) * snapshot_interval
    with open(f"data/snapshot_{t}.bin", "rb") as f:
        frames[i] = np.fromfile(f, dtype=np.float32).reshape(nx, ny)

# Create animation
fig, ax = plt.subplots()
cax = ax.imshow(frames[0], cmap='viridis', vmin=frames.min(), vmax=frames.max())
fig.colorbar(cax)

def animate(frame_index):
    cax.set_data(frames[frame_index])
    ax.set_title(f"Time step: {frame_index * snapshot_interval}")
    return [cax]

ani = animation.FuncAnimation(fig, animate, frames=range(frames.shape[0]), interval=100, blit=True)
ani.save("simulation.mp4", writer="ffmpeg", fps=10)
plt.close()