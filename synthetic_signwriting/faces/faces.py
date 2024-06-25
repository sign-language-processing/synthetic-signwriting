import math
import pickle
import numpy as np
from scipy.spatial.transform import Rotation


def get_face_rotation(pose: np.ndarray):
    point1 = pose[4]  # Nose
    point2 = pose[6]  # Middle eyebrows
    vec = point2 - point1

    return 90 + math.degrees(math.atan2(vec[1], vec[0]))


def rotate_face(pose: np.ndarray, angle: float):
    rotation = Rotation.from_euler('z', angle, degrees=True)
    return np.dot(pose, rotation.as_matrix())


def scale_face(pose: np.ndarray, suitable_face: np.ndarray = None, size=200):
    point1 = pose[4]  # Nose
    point2 = pose[6]  # Middle eyebrows
    current_size = np.sqrt(np.power(point2 - point1, 2).sum())

    if suitable_face is not None:
        point1 = suitable_face[4]  # Nose of suitable hand
        point2 = suitable_face[6]  # Middle eyebrows of suitable hand
        size = np.sqrt(np.power(point2 - point1, 2).sum())

    pose *= size / current_size
    pose -= pose[4] - suitable_face[4]  # move to Nose of the suitable hand
    return pose


def normalized_face(pose: np.ndarray):
    assert pose.shape == (468, 3)
    assert not np.all(pose == 0)

    # Then rotate on the X-Y plane such that the BASE-M_CMC is on the Y axis
    angle = get_face_rotation(pose)
    pose = rotate_face(pose, angle)

    return pose


def prorate_face(pose: np.ndarray, suitable_face: np.ndarray):
    assert pose.shape == (468, 3) and suitable_face.shape == (468, 3)
    assert not np.all(pose == 0) and not np.all(suitable_face == 0)
    pose = normalized_face(pose)

    # Scale pose such that BASE-M_CMC is of size of the suitable face
    pose = scale_face(pose, suitable_face)

    return pose


def read_pkl_file(file_path):
    try:
        with open(file_path, "rb") as file:
            data = pickle.load(file)
        return data
    except Exception as exp:    # pylint: disable=broad-except
        print(exp)
        return None
