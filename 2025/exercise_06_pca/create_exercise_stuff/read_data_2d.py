"""
prepare_airfoil_2d_binned.py

Fast binning version: avoids slow griddata interpolation.

Produces:
 - coords_full.npy         : original (n_points, 2)
 - mask_full.npy           : boolean mask for wing interior
 - velocities_grid.npz     : { 'mag': (nt, NY, NX), 'mean_mag': (NY, NX) or None,
                               'times': times, 'X': X_grid, 'Y': Y_grid, 'mask_grid': mask_grid }

Assumptions:
 - Unzipped snapshots in snapshots_midspan/
 - Snapshot names: airfoilLES_t00001.h5 ... airfoilLES_t16000.h5
 - Grid file: airfoilLES_grid.h5
 - Mean file (optional): airfoilLES_mean_midspan.h5
"""
import os
import glob
import h5py
import numpy as np
from matplotlib.path import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

# --- Nearest-neighbor NaN fill ---
def interpolate_2d_nearest(arr):
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError("Input array must be 2D.")
    mask = ~np.isnan(arr)
    if mask.all():
        return arr.copy()
    inds = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    filled = arr[tuple(inds)]
    return filled


# ---------- USER PARAMETERS ----------
GRID_FILE = "airfoilLES_grid.h5"
MEAN_FILE = "airfoilLES_mean_midspan.h5"   # optional
SNAPSHOT_DIR = "snapshots_midspan"
OUTDIR = "prepared_data_2d"
NT_MAX = None   # set e.g. 500 for fewer timesteps

# Fixed grid parameters
NX, NY = 1000, 500   # number of points in x and y directions
# -------------------------------------

os.makedirs(OUTDIR, exist_ok=True)

# ---- 1) load grid and airfoil geometry ----
print("Loading grid and airfoil geometry...")
with h5py.File(GRID_FILE, "r") as f:
    x = np.array(f["x"])
    y = np.array(f["y"])
    xa = np.array(f["xa"])
    ya = np.array(f["ya"])
coords = np.column_stack((x, y))
n_points = len(coords)
print(f" - Total grid points: {n_points}")

# ---- 2) build binary mask for airfoil ----
print("Building binary wing mask...")
poly = Path(np.column_stack((xa, ya)))
mask_full = poly.contains_points(coords)
np.save(os.path.join(OUTDIR, "coords_full.npy"), coords)
np.save(os.path.join(OUTDIR, "mask_full.npy"), mask_full)
print(" - Saved coords_full.npy and mask_full.npy")

# ---- 3) load mean flow (optional) ----
mean_vec = None
if os.path.exists(MEAN_FILE):
    with h5py.File(MEAN_FILE, "r") as fm:
        keys = list(fm.keys())
        ux_keys = [k for k in keys if "ux" in k and "mean" in k]
        uy_keys = [k for k in keys if "uy" in k and "mean" in k]
        if ux_keys and uy_keys:
            mean_ux = np.array(fm[ux_keys[0]]).ravel()
            mean_uy = np.array(fm[uy_keys[0]]).ravel()
            mean_vec = np.column_stack((mean_ux, mean_uy))
            print(" - Mean flow loaded from file.")
        else:
            print(" - Mean arrays not found; skipping mean.")
else:
    print("Mean file not found; skipping mean.")

# ---- 4) load snapshots ----
pattern = os.path.join(SNAPSHOT_DIR, "airfoilLES_t*.h5")
snap_files = sorted(glob.glob(pattern))
snap_files = snap_files[::2]
if NT_MAX is not None:
    snap_files = snap_files[:NT_MAX]
nt = len(snap_files)
if nt == 0:
    raise RuntimeError(f"No snapshot files found with pattern {pattern}")
print(f" - Found {nt} snapshots to load.")

data = np.zeros((nt, n_points, 2), dtype=np.float32)
for it, fn in enumerate(snap_files):
    with h5py.File(fn, "r") as fs:
        ux = np.array(fs["ux"]).ravel()
        uy = np.array(fs["uy"]).ravel()
        vec = np.column_stack((ux, uy))
        #if mean_vec is not None:
        #    vec -= mean_vec   # subtract LES mean if desired
        data[it] = vec
    if (it + 1) % 50 == 0 or it == nt - 1:
        print(f"   Loaded {it + 1}/{nt} snapshots")

times = np.arange(nt, dtype=np.float32)

# ---- 5) define grid edges and indices ----
x_edges = np.linspace(coords[:,0].min(), coords[:,0].max(), NX+1)
y_edges = np.linspace(coords[:,1].min(), coords[:,1].max(), NY+1)
x_idx = np.digitize(coords[:,0], x_edges) - 1
y_idx = np.digitize(coords[:,1], y_edges) - 1
x_idx = np.clip(x_idx, 0, NX-1)
y_idx = np.clip(y_idx, 0, NY-1)


# ---- 6) bin points and compute mean magnitude per cell ----
mag = data #np.linalg.norm(data, axis=-1)  # (nt, n_points) # unclear if this is the right averaging!
mean_mag = mean_vec #np.linalg.norm(mean_vec, axis=-1) #(n_points)
mag_grid = np.full((nt, NY, NX, 2), np.nan, dtype=np.float32)
mean_mag_grid = np.full((NY, NX, 2), np.nan, dtype=np.float32)
for idx in tqdm(range(NY * NX), desc="Cells"):
    i = idx // NX
    j = idx % NX
    mask = (y_idx == i) & (x_idx == j)
    if np.any(mask):
        mag_grid[:, i, j, :] = mag[:, mask, :].mean(axis=1)
        mean_mag_grid[i, j, :] = mean_mag[mask, :].mean(axis=0)
        
mag_grid = np.linalg.norm(mag_grid, axis=-1)
mean_mag_grid = np.linalg.norm(mean_mag_grid, axis=-1)      

# ---- 8) create meshgrid for plotting ----
X_grid, Y_grid = np.meshgrid(
    0.5*(x_edges[:-1]+x_edges[1:]),
    0.5*(y_edges[:-1]+y_edges[1:])
)

# ---- 9) apply airfoil mask ----
mask_grid = poly.contains_points(np.column_stack([X_grid.ravel(), Y_grid.ravel()])).reshape(NY, NX)
mag_grid[:, mask_grid] = np.nan
if mean_mag_grid is not None:
    mean_mag_grid[mask_grid] = np.nan
    

    
for i in tqdm(range(mag_grid.shape[0]),desc="Interpolation"):
    mag_grid[i] = interpolate_2d_nearest(mag_grid[i])

# ---- 10) save prepared data ----
out_npz_grid = os.path.join(OUTDIR, "velocities_grid_binned.npz")
np.savez_compressed(
    out_npz_grid,
    mag=mag_grid,
    mean_mag=mean_mag_grid,
    times=times,
    X=X_grid,
    Y=Y_grid,
    mask_grid=mask_grid
)
print(f"Saved binned grid data to {out_npz_grid}")
 