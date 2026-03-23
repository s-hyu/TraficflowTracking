from pathlib import Path

import cv2
import numpy as np

base = Path(__file__).resolve().parent
pair_file = base / "point_pairs.txt"

# point_pairs.txt format per line:
# left_x left_y center_x center_y
pairs = np.loadtxt(str(pair_file), dtype=np.float64)

# Use only first 30 pairs for fitting.
train_pairs = pairs[:30]
left_pts = train_pairs[:, 0:2]
center_pts = train_pairs[:, 2:4]

H, _ = cv2.findHomography(left_pts, center_pts, 0)

np.savetxt(str(base / "H_left_to_center.txt"), H, fmt="%.10f")
np.savetxt(str(base / "H_center_to_left.txt"), np.linalg.inv(H), fmt="%.10f")

print("Used pairs:", len(train_pairs))
print("H (left -> center):")
print(H)
print("Saved:", base / "H_left_to_center.txt")
