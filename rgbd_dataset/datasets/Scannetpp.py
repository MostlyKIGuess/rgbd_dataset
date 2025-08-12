from pathlib import Path
import glob
import numpy as np
from typing import List
from natsort import natsorted
import json
from scipy.spatial.transform import Rotation
from numpy.typing import NDArray


from ..BaseRGBDDataset import BaseRGBDDataset
from ..utils import invert_se3
from .nerfstudio_utils import load_only_transforms


# This uses the ARK Kit Pose. See the other class for the COLMAP pose
class Scannetpp(BaseRGBDDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_resized = (
            True  # Always resize because of depth has much lower resolution
        )

    def get_rgb_paths(self) -> List[str]:
        path_str = str(
            self.base_path / "data" / self.scene / "iphone" / "rgb" / "frame*.jpg"
        )
        rgb_paths = natsorted(glob.glob(path_str))
        return rgb_paths

    def get_depth_paths(self) -> List[str]:
        path_str = str(
            self.base_path / "data" / self.scene / "iphone" / "depth" / "frame*.png"
        )
        depth_paths = natsorted(glob.glob(path_str))
        return depth_paths

    def get_se3_poses(self) -> List[np.array]:
        pose_path = str(
            self.base_path / "data" / self.scene / "iphone" / "pose_intrinsic_imu.json"
        )
        with open(pose_path, "r") as f:
            data = json.load(f)
        keys = natsorted([k for k in data.keys()])

        poses = [np.array(data[k]["aligned_pose"]).reshape((4, 4)) for k in keys]

        return poses

    def get_intrinsic_matrices(self) -> List[np.array]:
        pose_path = str(
            self.base_path / "data" / self.scene / "iphone" / "pose_intrinsic_imu.json"
        )
        with open(pose_path, "r") as f:
            data = json.load(f)
        keys = natsorted([k for k in data.keys()])

        poses = [np.array(data[k]["intrinsic"]) for k in keys]

        return poses


class COLMAP_image_txt:
    def __init__(self, path):
        self.file = open(str(path), "r")

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self.file).split()
        while not len(line) or line[0] == "#" or not line[-1].endswith(".jpg"):
            line = next(self.file).split()

        # Some scenes have a leading video/ in the path for no apparent reason
        line[-1] = line[-1].replace("video/", "")

        return line


class ScannetppCOLMAP(BaseRGBDDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_resized = (
            True  # Always resize because of depth has much lower resolution
        )

    def get_rgb_paths(self) -> List[str]:
        rgb_paths = []
        lines = COLMAP_image_txt(
            self.base_path / "data" / self.scene / "iphone" / "colmap" / "images.txt"
        )
        for line in lines:
            rgb_paths.append(
                self.base_path / "data" / self.scene / "iphone" / "rgb" / line[-1]
            )

        return rgb_paths

    def get_depth_paths(self) -> List[str]:
        depth_paths = []
        lines = COLMAP_image_txt(
            self.base_path / "data" / self.scene / "iphone" / "colmap" / "images.txt"
        )
        for line in lines:
            file_name = line[-1].replace("jpg", "png")
            depth_paths.append(
                self.base_path / "data" / self.scene / "iphone" / "depth" / file_name
            )

        return depth_paths

    def get_se3_poses(self) -> List[np.array]:
        se3_poses = []
        lines = COLMAP_image_txt(
            self.base_path / "data" / self.scene / "iphone" / "colmap" / "images.txt"
        )
        for line in lines:
            quat = np.array(line[1:5]).astype(float)
            translation = np.array(line[5:8]).astype(float)
            se3 = np.eye(4)
            se3[:3, :3] = Rotation.from_quat(quat, scalar_first=True).as_matrix()
            se3[:3, 3] = translation
            se3_poses.append(invert_se3(se3))

        return se3_poses


class ScannetppDSLR(BaseRGBDDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_resized = (
            True  # Always resize because of depth has much lower resolution
        )

    def get_rgb_paths(self) -> List[str]:
        base = self.base_path / "data" / self.scene / "dslr"
        candidates = []
        # for folder in ["resized_images" ]:
        for folder in ["undistorted_images"]:
            for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]:
                path_str = str(base / folder / ext)
                found = glob.glob(path_str)
                if found:
                    print(
                        f"[ScannetppDSLR] Found {len(found)} images in {folder} with extension {ext}"
                    )
                    candidates.extend(found)
        if not candidates:
            print(
                "[ScannetppDSLR] No images found in either resized_images/ or resized_undistorted_images/ with jpg/jpeg extensions!"
            )
        rgb_paths = natsorted(candidates)
        return rgb_paths

    def get_depth_paths(self) -> List[str]:
        # Return sorted list of .png files from render_depth/ to match RGB order
        base = self.base_path / "data" / self.scene / "dslr"
        print("#" * 50)
        print("[ScannetppDSLR] Looking for depth images in:")
        print(base)
        depth_paths = natsorted(
            glob.glob(str(base / "render_depth_undistorted" / "*.png"))
        )
        print(depth_paths)
        print("#" * 40)
        if not depth_paths:
            print("[ScannetppDSLR] No depth images found in render_depth/!")
        return depth_paths

    def get_se3_poses(self) -> List[NDArray]:
        from pathlib import Path

        pose_data_file = (
            self.base_path
            / "data"
            / self.scene
            / "dslr"
            / "nerfstudio"
            / "transforms_undistorted.json"
        )
        frames = load_only_transforms(pose_data_file)

        def normalize_img_name(name):
            return Path(name).name.lower().replace(".jpeg", ".jpg")

        frames_norm = {normalize_img_name(k): v for k, v in frames.items()}
        rgb_paths = self.get_rgb_paths()
        rgb_basenames = [normalize_img_name(p) for p in rgb_paths]
        ordered_poses = []
        missing = []
        OPENGL_TO_OPENCV = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        for fname in rgb_basenames:
            if fname not in frames_norm:
                missing.append(fname)
                continue
            transform = np.array(frames_norm[fname], dtype=np.float32)
            pose_cv = OPENGL_TO_OPENCV @ transform @ OPENGL_TO_OPENCV
            ordered_poses.append(pose_cv)
        if missing:
            print(
                f"[ScannetppDSLR] Warning: {len(missing)} RGB files not found in Nerfstudio frames. Examples: {missing[:5]}"
            )
            print(
                f"Available Nerfstudio frame keys (first 5): {list(frames_norm.keys())[:5]}"
            )
        if not ordered_poses:
            raise RuntimeError(
                "No matching poses found for any RGB image. Check your data consistency and filename normalization."
            )
        return ordered_poses

    def get_intrinsic_matrices(self) -> List[NDArray]:
        camera_intrinsics = {}
        import json

        nerfstudio_json_path = (
            self.base_path
            / "data"
            / self.scene
            / "dslr"
            / "nerfstudio"
            / "transforms_undistorted.json"
        )
        with open(nerfstudio_json_path, "r") as f:
            data = json.load(f)
        fl_x = data["fl_x"]
        fl_y = data["fl_y"]
        cx = data["cx"]
        cy = data["cy"]
        K = np.array(
            [
                [fl_x, 0, cx],
                [0, fl_y, cy],
                [0, 0, 1],
            ]
        )
        colmap_cameras_path = (
            self.base_path / "data" / self.scene / "dslr" / "colmap" / "cameras.txt"
        )
        with open(colmap_cameras_path, "r") as f:
            for line in f:
                if line.startswith("#") or len(line.strip()) == 0:
                    continue
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                camera_id = parts[0]
                camera_intrinsics[camera_id] = K

        # Use the same K for all camera IDs
        # for per-image intrinsics, modify here
        # For now, assign to all camera IDs found in images.txt
        colmap_images_path = (
            self.base_path / "data" / self.scene / "dslr" / "colmap" / "images.txt"
        )
        # Parse images.txt to get camera_id for each image (only metadata lines)
        image_camera_ids = []
        image_names = []
        with open(colmap_images_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith("#") or len(line.strip()) == 0:
                continue
            parts = line.strip().split()
            # Only process lines with a .jpg/.jpeg filename in the expected position
            if len(parts) >= 10 and (
                parts[9].lower().endswith(".jpg") or parts[9].lower().endswith(".jpeg")
            ):
                camera_id = parts[8]
                image_name = normalize_img_name(parts[9])
                image_camera_ids.append(camera_id)
                image_names.append(image_name)
        rgb_paths = self.get_rgb_paths()
        rgb_names = [normalize_img_name(p) for p in rgb_paths]
        print(
            "[ScannetppDSLR] Found {} images in resized_images/ folder.".format(
                len(rgb_names)
            )
        )
        print("[ScannetppDSLR] First 10 rgb_names:", rgb_names[:10])
        print("[ScannetppDSLR] First 10 image_names from COLMAP:", image_names[:10])
        name_to_K = {
            name: camera_intrinsics[cid]
            for name, cid in zip(image_names, image_camera_ids)
            if cid in camera_intrinsics
        }
        ordered_K = []
        missing = []
        for key in rgb_names:
            if key in name_to_K:
                ordered_K.append(name_to_K[key])
            else:
                missing.append(key)
        if missing:
            print(
                f"[ScannetppDSLR] Warning: {len(missing)} images in folder not found in COLMAP images.txt: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        if not ordered_K:
            raise RuntimeError(
                "No matching intrinsics found for any RGB image. Check your data consistency."
            )
        return ordered_K


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion [w, x, y, z] to a 3x3 rotation matrix.
    """
    quat_scipy = np.roll(qvec, -1)  # Convert to [x, y, z, w]
    rot = Rotation.from_quat(quat_scipy)
    return rot.as_matrix()


def normalize_img_name(name):
    """Normalize image filename for robust matching (lowercase, .jpeg->.jpg)."""
    return Path(name).name.lower().replace(".jpeg", ".jpg")
