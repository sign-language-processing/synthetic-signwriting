from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy import ma
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeaderDimensions, PoseHeader
from pose_format.utils.holistic import holistic_hand_component
from pose_format.utils.normalization_3d import PoseNormalizer


def hands_to_pose(hand: np.ndarray) -> Pose:
    header = PoseHeader(version=0.1,
                        dimensions=PoseHeaderDimensions(width=1, height=1, depth=1),
                        components=[holistic_hand_component("RIGHT_HAND_LANDMARKS")])

    hand = hand.reshape((-1, 1, 21, 3))
    confidence_shape = hand.shape[:-1]
    body = NumPyPoseBody(fps=1, data=hand, confidence=np.ones(confidence_shape))

    return Pose(header, body)


@lru_cache()
def get_hand_normalizer():
    dummy_pose = hands_to_pose(np.zeros((21, 3)))

    plane = dummy_pose.header.normalization_info(p1=("RIGHT_HAND_LANDMARKS", "WRIST"),
                                                 p2=("RIGHT_HAND_LANDMARKS", "PINKY_MCP"),
                                                 p3=("RIGHT_HAND_LANDMARKS", "INDEX_FINGER_MCP"))
    line = dummy_pose.header.normalization_info(p1=("RIGHT_HAND_LANDMARKS", "WRIST"),
                                                p2=("RIGHT_HAND_LANDMARKS", "MIDDLE_FINGER_MCP"))
    return PoseNormalizer(plane=plane, line=line, size=150)


@lru_cache(maxsize=1)
def load_hands():
    file_path = Path(__file__).parent / "hands.npy"
    return np.load(file_path)


@lru_cache(maxsize=1)
def load_hands_3d():
    # Load relevant hand crops and views
    hands = load_hands()
    hands = hands[16:32]  # only get "good" hand crops
    hands = hands[:, :, :3]  # only get 3 wall plane views

    # Normalize hands
    hands_shape = hands.shape
    hands = hands.reshape((-1, 1, 21, 3))
    normalizer = get_hand_normalizer()
    normalized_hands = normalizer(ma.masked_array(hands))
    hands = normalized_hands.reshape(hands_shape)

    # Get hands median
    hands = hands.transpose((0, 2, 1, 3, 4))
    hands = ma.concatenate(hands, axis=0)
    hands = ma.median(hands, axis=0)

    return hands


def get_hand_signwriting_symbol(hand_index: int) -> int:
    fsw_hex = (0x100 + hand_index) * 0x100
    if hand_index in [77, 79, 81, 92, 94, 246, 260]:
        fsw_hex += 0x10
    return fsw_hex


def get_hand_signwriting(hand_index: int) -> str:
    symbol = get_hand_signwriting_symbol(hand_index)
    symbol_hex = "S" + hex(symbol)[2:]
    return f"M508x515{symbol_hex}493x485"
