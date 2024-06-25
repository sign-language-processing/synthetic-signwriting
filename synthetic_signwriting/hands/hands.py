import math
import numpy as np

from scipy.spatial.transform import Rotation


def create_left_hand(pose: np.ndarray):
    min_p = np.min(pose, axis=0)  # Get the maximum point in the pose

    # multiply the x-axis by -1 to reflect the hand
    left_hand = pose
    left_hand[:, 0] = -left_hand[:, 0]

    diff = min_p - np.min(left_hand, axis=0)
    left_hand += diff

    return left_hand


def get_hand_rotation(pose: np.ndarray):
    point_1 = pose[0]  # Wrist
    point_2 = pose[9]  # Middle CMC
    vec = point_2 - point_1

    return 90 + math.degrees(math.atan2(vec[1], vec[0]))


def rotate_hand(pose: np.ndarray, angle: float, axis='z'):
    rotation = Rotation.from_euler(axis, angle, degrees=True)
    return np.dot(pose, rotation.as_matrix())


def scale_hand(pose: np.ndarray, suitable_hand: np.ndarray = None, size=200):
    point_1 = pose[0]  # Wrist
    point_2 = pose[9]  # Middle CMC
    current_size = np.sqrt(np.power(point_2 - point_1, 2).sum())
    if suitable_hand is not None:
        point_1 = suitable_hand[0]  # Wrist of suitable hand
        point_2 = suitable_hand[9]  # Middle CMC of suitable hand
        size = np.sqrt(np.power(point_2 - point_1, 2).sum())
    pose *= size / current_size
    pose += suitable_hand[0] - pose[0]  # move to Wrist of the suitable hand
    return pose


def prorate_hand(pose: np.ndarray, suitable_hand: np.ndarray, reflection=False):
    assert pose.shape == (21, 3) and suitable_hand.shape == (21, 3)
    assert not np.all(pose == 0) and not np.all(suitable_hand == 0)

    pose = pose.copy()

    if reflection:
        pose = create_left_hand(pose)

    # Scale pose such that BASE-M_CMC is of size of the suitable hand
    pose = scale_hand(pose, suitable_hand)

    return pose


def read_npy_file(file_path):
    try:
        data = np.load(file_path)
        return data
    except Exception as exp:  # pylint: disable=broad-except
        print(f"An error occurred while reading the file: {exp}")
        return None
