import json
from pathlib import Path
import numpy as np

def load_only_transforms(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return {Path(frame["file_path"]).name: np.array(frame["transform_matrix"], dtype=np.float32) for frame in data["frames"]}

def check_duplicates(seq):
    seen = set()
    dups = set()
    for x in seq:
        if x in seen:
            dups.add(x)
        seen.add(x)
    return list(dups)

OPENGL_TO_OPENCV = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32
)

def convert_pose_opengl_to_opencv(transform):
    return OPENGL_TO_OPENCV @ transform @ OPENGL_TO_OPENCV
