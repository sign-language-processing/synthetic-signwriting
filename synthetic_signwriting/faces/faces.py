import math
import pickle
from functools import lru_cache
from pathlib import Path
from pose_format.utils.normalization_3d import PoseNormalizer, PoseNormalizationInfo
import numpy as np
import numpy.ma as ma
from scipy.spatial.transform import Rotation


def rotate_face(pose: np.ndarray, angle: float, axis='z'):
    rotation = Rotation.from_euler(axis, angle, degrees=True)
    return np.dot(pose, rotation.as_matrix())


def reposition_face(pose: np.ndarray, suitable_face: np.ndarray):
    pose_center_point = pose[4]  # Nose
    suitable_face_center_point = suitable_face[4]  # Nose of suitable hand
    pose += suitable_face_center_point - pose_center_point  # move to Nose of the suitable hand
    return pose


def proportionate_face(pose: np.ndarray, suitable_face: np.ndarray):
    assert pose.shape == (468, 3) and suitable_face.shape == (468, 3)
    assert not np.all(pose == 0) and not np.all(suitable_face == 0)

    # reposition face to the center of the suitable face
    pose = reposition_face(pose, suitable_face)

    return pose


def normalize_face(normalizer: PoseNormalizer, pose: np.ndarray):
    pose = normalizer(ma.masked_array([[pose]]))[0][0]
    pose = rotate_face(pose, -50, 'x')  # Rotate on the x axis by 45 degrees
    return pose


def face_normalization():
    plane = PoseNormalizationInfo(4, 362, 133)  # Nose, Right eye inner, Left eye inner
    line = PoseNormalizationInfo(4, 6)  # Nose, Middle eyebrows
    return PoseNormalizer(plane, line, 45)


@lru_cache()
def load_faces():
    file_path = Path(__file__).parent.parent / "data" / "faces.pkl"
    try:
        with open(file_path, "rb") as file:
            data = pickle.load(file)
        return data
    except Exception as exp:  # pylint: disable=broad-except
        print(exp)
        return None
