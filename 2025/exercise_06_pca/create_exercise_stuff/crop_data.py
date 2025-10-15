
import numpy as np
# --- Load binned 2D data ---
data = np.load("prepared_data_2d/velocities_grid_binned_cropped.npz")
mag_grid = data["mag"]             # (nt, NY, NX)
mean_mag_grid = data["mean_mag"]   # (NY, NX) or None
X = data["X"]
Y = data["Y"]
mask_grid = data["mask_grid"]
times = data["times"]



# ---- 11) crop to region of interest and save ----
# Define limits
x_min, x_max = -0.25, 4
y_min, y_max = -1, 1

# Find indices inside limits
ix = np.where((X >= x_min) & (X <= x_max))[0]
iy = np.where((Y >= y_min) & (Y <= y_max))[0]

# Crop all arrays
mag_grid_crop = mag_grid[:, iy.min():iy.max()+1, ix.min():ix.max()+1]
mask_grid_crop = mask_grid[iy.min():iy.max()+1, ix.min():ix.max()+1]
X_crop = X[ix.min():ix.max()+1]
Y_crop = Y[iy.min():iy.max()+1]

if mean_mag_grid is not None:
    mean_mag_grid_crop = mean_mag_grid[iy.min():iy.max()+1, ix.min():ix.max()+1]
else:
    mean_mag_grid_crop = None

# Save cropped data
np.savez(
    "prepared_data_2d/velocities_grid_binned_cropped.npz",
    mag=mag_grid_crop,
    mean_mag=mean_mag_grid_crop,
    X=X_crop,
    Y=Y_crop,
    mask_grid=mask_grid_crop,
    times=times,
)

print("Saved cropped data:",
      mag_grid_crop.shape, "grid size", len(X_crop), "x", len(Y_crop))
print("Done.")
