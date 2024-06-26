import random

import numpy as np
from pose_format import Pose

from synthetic_signwriting.hands.hands import read_npy_file, prorate_hand
from synthetic_signwriting.faces.faces import prorate_face, read_pkl_file
from synthetic_signwriting.synthetic_pose_helper import randomize_hand_pose, create_base_frame, pose_transition


class SyntheticSignWriting:

    def __init__(self):
        with open('data/base_pose.pose', 'rb') as file:
            data_buffer = file.read()
        pose = Pose.read(data_buffer)
        middle = int(len(pose.body.data) / 2)

        self.hand_data = read_npy_file('data/hands.npy')
        self.face_data = read_pkl_file('data/faces.pkl')
        self.pose = pose
        self.static_position = [pose.body.data[middle + 2 * i][0].copy() for i in range(4)]
        self.hands = []
        self.faces = []

    def add_keyframe(self, face=None, left_hand=None, right_hand=None):
        # Create and configure a new frame
        self.hands.append((left_hand, right_hand))
        self.faces.append(face)

    def add_random_keyframe(self):
        # choose a random hand and face pose
        right_hand = randomize_hand_pose(self.hand_data)  # choose a random hand pose
        left_hand = randomize_hand_pose(self.hand_data)  # choose a random hand pose

        # choose a random face pose
        face_type = random.choice(list(self.face_data.keys()))
        element = random.choice(list(self.face_data[face_type].keys()))
        if element == "source":
            face = self.face_data[face_type][element]
        else:
            cathartic = random.choice(list(self.face_data[face_type][element].keys()))
            face = self.face_data[face_type][element][cathartic]

        # add the keyframe
        self.add_keyframe(face=face, left_hand=left_hand, right_hand=right_hand)

    def render_signwriting(self):
        # Use the list of hands and faces to generate SignWriting
        return "..."

    def render_pose(self, frame_suspension=100):
        self.pose.body.data = None
        zero_frame = np.zeros((frame_suspension - 2, 1, 543, 3))  # the -2 is to remove the first and last frame

        for face, (left_hand, right_hand) in zip(self.faces, self.hands):
            selected_frame = random.choice(self.static_position)  # choose a random base frame
            pose_frame = self.static_position[selected_frame][0].copy()

            # hand
            right_hand_base = pose_frame[522:543]  # get the right hand base that located (522:543)
            right_hand = prorate_hand(right_hand, right_hand_base)
            pose_frame[522:543] = right_hand
            left_hand_base = pose_frame[501:522]  # get the left hand base that located (501:522)
            left_hand = prorate_hand(left_hand, left_hand_base, reflection=True)  # reflection because it's left hand
            pose_frame[501:522] = left_hand

            # face
            face_base = pose_frame[33:501]  # get the face base that located (33:501)
            face = prorate_face(face, face_base)
            pose_frame[33:501] = face

            if self.pose.body.data is None:
                self.pose.body.data = [[create_base_frame(pose_frame)]]
            self.pose.body.data += zero_frame.copy()
            self.pose.body.data.append([pose_frame])
        self._smooth(frame_suspension)
        return self.pose

    def _smooth(self, frame_suspension):
        suspension_on_pose = 30  # adding frames of suspension after each keyframe to emphasize the target pose
        for i in range(0, len(self.pose.body.data) - 1, frame_suspension - 1):
            self.pose.body.data[i: frame_suspension] = pose_transition(self.pose.body.data[i],
                                                                       self.pose.body.data[i + frame_suspension - 1],
                                                                       frame_suspension - suspension_on_pose,
                                                                       suspension_on_pose)
