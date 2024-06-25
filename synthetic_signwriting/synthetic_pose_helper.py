import random
from pose_format import Pose
import numpy as np

from synthetic_signwriting.hands.hands import read_npy_file, prorate_hand, get_hand_rotation, rotate_hand


def randomize_hand_pose(pose: np.ndarray):
    matrix_type = random.choice([0, 1])
    shape = random.choice(range(261))
    orientations = random.choice(range(6))
    return pose[matrix_type][shape][orientations]


def create_base_frame(pose: Pose, next_pose: np.ndarray = None):
    open_hand = 77  # simple hand that is open
    starting_pose = pose.body.data[0][0].copy()  # get the first frame
    hand_data_path = 'data/hands.npy'
    hand_data = read_npy_file(hand_data_path)
    matrix_type = random.choice([0, 1])

    # rotate the hands to be in the right position
    basic_hand = hand_data[matrix_type][open_hand][0]
    basic_hand = rotate_hand(basic_hand, get_hand_rotation(starting_pose) + 180)
    basic_hand = rotate_hand(basic_hand, 90, 'x')  # rotate the hand to rest position

    next_right_hand = next_pose[522:543]  # get the right hand base that located (522:543)
    right_hand = prorate_hand(basic_hand, next_right_hand)
    right_wrist = starting_pose[16]  # 16 is the right wrist
    right_hand += right_wrist - right_hand[0]  # normalize the right hand

    next_left_hand = next_pose[501:522]  # get the left hand base that located (501:522)
    left_hand = prorate_hand(basic_hand, next_left_hand, reflection=True)
    left_wrist = starting_pose[15]  # 15 is the left wrist
    left_hand += left_wrist - left_hand[0]  # normalize the left hand

    # update the right and left hand
    starting_pose[522:543] = right_hand
    starting_pose[501:522] = left_hand

    return starting_pose


def pose_transition(begin_position: np.ndarray, end_position: np.ndarray, frames: int, buffer: int = 30):
    assert begin_position.shape == end_position.shape
    transition = np.zeros((frames + buffer, *begin_position.shape))
    step = (end_position - begin_position) / frames
    transition[0] = begin_position
    for i in range(1, frames):
        #   go one step forward
        transition[i] = transition[i - 1] + step

        #   normalize the right hand based on the right wrist
        right_hand = transition[i][522:543]  # 522:543 is the right hand
        right_wrist = transition[i][16]  # 16 is the right wrist
        right_hand += right_wrist - right_hand[0]  # normalize the right hand

        left_hand = transition[i][501:522]  # 501:522 is the left hand
        left_wrist = transition[i][15]  # 15 is the left wrist
        left_hand += left_wrist - left_hand[0]  # normalize the left hand

        #  update the right and left hand
        transition[i][522:543] = right_hand
        transition[i][501:522] = left_hand
        step = (end_position - transition[i]) / (frames - i)

    # add suspension frames
    for i in range(frames, frames + buffer):
        transition[i] = end_position

    transition = np.ma.masked_array([[frame] for frame in transition])
    return transition
