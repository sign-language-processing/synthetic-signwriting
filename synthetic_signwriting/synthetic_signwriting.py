import random

from pose_format import Pose


class SyntheticSignWriting:
    pose = Pose(header=None, body=None)

    def __init__(self):
        self.key_frames = []
        self.hands = []
        self.faces = []

    def add_keyframe(self, face=True, left_hand=True, right_hand=True):
        # Create and configure a new frame
        frame = self.pose.create_frame()
        if face:
            frame.set_face(...)
        if left_hand:
            frame.set_left_hand(...)
        if right_hand:
            frame.set_right_hand(...)

    def render_signwriting(self):
        # Use the list of hands and faces to generate SignWriting
        return "..."

    def render_pose(self):
        self._smooth()
        # Apply camera transformation and render the pose sequence
        return "..."

    def _smooth(self, kind=None):
        if kind is None:
            kind = random.choice(["linear", "quadratic", "cubic"])

        self.pose = self.pose.interpolate(kind=kind)
