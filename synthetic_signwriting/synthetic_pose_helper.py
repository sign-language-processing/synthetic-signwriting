import random
from pose_format import Pose
import numpy as np

from synthetic_signwriting.hands.hands import load_hands, normalize_hand, hands_normalization, create_left_hand, rotate_hand
from synthetic_signwriting.faces.faces import normalize_face, face_normalization, proportionate_face

RightHand_Range = slice(522, 543)  # 522:543 is the right hand
LeftHand_Range = slice(501, 522)  # 501:522 is the left hand
Head_Range = slice(33, 501)  # 33:501 is the head


def create_base_frame(starting_pose: np.ndarray):
    open_hand = 77  # simple hand that is open
    hand_data = load_hands()
    matrix_type = random.choice([0, 1])
    hand_normalizer = hands_normalization()
    face_normalizer = face_normalization()

    # rotate the hands to be in the right position
    basic_hand = hand_data[matrix_type][open_hand][0]
    basic_hand = normalize_hand(hand_normalizer, basic_hand)
    basic_hand = rotate_hand(basic_hand, 180, 'z')  # rotate the hand to rest position

    right_hand = basic_hand.copy()
    right_wrist = starting_pose[16]  # 16 is the right wrist
    right_hand += right_wrist - right_hand[0]  # normalize the right hand

    left_hand = create_left_hand(basic_hand)
    left_wrist = starting_pose[15]  # 15 is the left wrist
    left_hand += left_wrist - left_hand[0]  # normalize the left hand

    normalized_face = normalize_face(face_normalizer, starting_pose[Head_Range].copy())     # normalize the face
    normalized_face = proportionate_face(normalized_face, starting_pose[Head_Range])        # proportionate the face

    # update the right and left hand
    starting_pose[RightHand_Range] = right_hand
    starting_pose[LeftHand_Range] = left_hand
    starting_pose[Head_Range] = normalized_face

    return starting_pose


def pose_transition(begin_position: np.ndarray, end_position: np.ndarray, frames: int, buffer: int = 30):
    assert begin_position.shape == end_position.shape
    begin_position = begin_position[0]
    end_position = end_position[0]

    transition = np.zeros((frames + buffer, *begin_position.shape))
    step = (end_position - begin_position) / frames
    transition[0] = begin_position

    for i in range(1, frames):
        #   go one step forward
        transition[i] = transition[i - 1] + step

        #   normalize the right hand based on the right wrist
        right_hand = transition[i][RightHand_Range]  # retrieve the right hand
        right_wrist = transition[i][16]  # 16 is the right wrist
        right_hand += right_wrist - right_hand[0]  # normalize the right hand

        left_hand = transition[i][LeftHand_Range]  # retrieve the left hand
        left_wrist = transition[i][15]  # 15 is the left wrist
        left_hand += left_wrist - left_hand[0]  # normalize the left hand

        #  update the right and left hand
        transition[i][RightHand_Range] = right_hand
        transition[i][LeftHand_Range] = left_hand
        step = (end_position - transition[i]) / (frames - i)

    # add suspension frames
    for i in range(frames, frames + buffer):
        transition[i] = end_position

    transition = np.ma.masked_array([[frame] for frame in transition])
    return transition
