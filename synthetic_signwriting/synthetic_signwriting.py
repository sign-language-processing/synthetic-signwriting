import random
from pathlib import Path

import numpy as np
from pose_format import Pose

from synthetic_signwriting.hands.hands import load_hands, proportionate_hand, randomize_hand_pose, hands_normalization, \
    normalize_hand
from synthetic_signwriting.faces.faces import proportionate_face, load_faces, face_normalization, normalize_face
from synthetic_signwriting.synthetic_pose_helper import create_base_frame, pose_transition


class SyntheticSignWriting:

    def __init__(self):
        base_pose_path = Path(__file__).parent / "data" / "base_pose.pose"
        with open(base_pose_path, 'rb') as file:
            pose = Pose.read(file.read())
        middle_frame = int(len(pose.body.data) / 2)

        # load the hand and face data
        self.hand_data = load_hands()
        self.face_data = load_faces()

        # initialize the pose
        self.pose = pose
        # initialize static position based on the base pose - will use as a base for the keyframes
        self.static_position = [pose.body.data[middle_frame + 2 * i][0].copy() for i in range(4)]
        self.start_position = pose.body.data[0][0].copy()  # the first frame
        self.sequence = []

        # initialize the hands and faces locations
        self.feature_range = {}
        offset = 0  # offset to keep track of the current location
        for component in self.pose.header.components:
            if component.name == "RIGHT_HAND_LANDMARKS":
                self.feature_range["right_hand"] = slice(offset, offset + len(component.points))
            if component.name == "LEFT_HAND_LANDMARKS":
                self.feature_range["left_hand"] = slice(offset, offset + len(component.points))
            if component.name == "FACE_LANDMARKS":
                self.feature_range["face"] = slice(offset, offset + len(component.points))
            offset += len(component.points)

    def add_keyframe(self, face=None, left_hand=None, right_hand=None):
        # Create and configure a new frame
        self.sequence.append((face, left_hand, right_hand))

    def add_random_keyframe(self):
        # choose a random hand and face pose
        right_hand = randomize_hand_pose(self.hand_data)
        left_hand = randomize_hand_pose(self.hand_data)

        # choose a random face pose
        face_type = random.choice(list(self.face_data.keys()))
        element = random.choice(list(self.face_data[face_type].keys()))
        if element == "source":
            face = self.face_data[face_type][element]
        else:
            characteristic = random.choice(list(self.face_data[face_type][element].keys()))
            face = self.face_data[face_type][element][characteristic]

        # add the keyframe
        self.add_keyframe(face=face, left_hand=left_hand, right_hand=right_hand)

    def render_signwriting(self):
        # Use the list of hands and faces to generate SignWriting
        return "..."

    def render_pose(self, frame_suspension=100):
        # initialize the pose
        _, _, points, dimensions = self.pose.body.data.shape  # get the number of points and dimensions
        self.pose.body.data = None
        zero_frame = np.zeros((frame_suspension - 2, 1, points, dimensions))  # the -2 is because first and last frame
        hands_normalizer = hands_normalization()
        face_normalizer = face_normalization()

        for (face, left_hand, right_hand) in self.sequence:
            selected_frame = random.choice(self.static_position)  # choose a random base frame
            pose_frame = selected_frame.copy()

            # hand
            right_hand_base = pose_frame[self.feature_range["right_hand"]]  # get the right hand
            right_hand = normalize_hand(hands_normalizer, right_hand)  # normalize the right hand
            right_hand = proportionate_hand(right_hand, right_hand_base)
            pose_frame[self.feature_range["right_hand"]] = right_hand

            left_hand_base = pose_frame[self.feature_range["left_hand"]]  # get the left hand
            left_hand = normalize_hand(hands_normalizer, left_hand)
            left_hand = proportionate_hand(left_hand, left_hand_base,
                                           reflection=True)  # reflection because it's left hand
            pose_frame[self.feature_range["left_hand"]] = left_hand

            # face
            face_base = pose_frame[self.feature_range["face"]]  # get the face
            face = normalize_face(face_normalizer, face)
            face = proportionate_face(face, face_base)
            pose_frame[self.feature_range["face"]] = face

            if self.pose.body.data is None:
                self.pose.body.data = np.array([[create_base_frame(self.start_position)]])
            self.pose.body.data = np.concatenate((self.pose.body.data, zero_frame.copy()))
            self.pose.body.data = np.concatenate((self.pose.body.data, np.array([[pose_frame]])))
        self._smooth(frame_suspension)
        return self.pose

    def _smooth(self, frame_suspension):
        suspension_on_pose = 30  # adding frames of suspension after each keyframe to emphasize the target pose
        for i in range(0, len(self.pose.body.data) - 1, frame_suspension - 1):
            self.pose.body.data[i: frame_suspension + i] = pose_transition(self.pose.body.data[i],
                                                                           self.pose.body.data[
                                                                               i + frame_suspension - 1],
                                                                           frame_suspension - suspension_on_pose,
                                                                           suspension_on_pose)
