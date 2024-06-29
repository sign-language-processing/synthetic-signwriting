import random
from typing import NamedTuple

import numpy as np
from pose_anonymization.appearance import get_mean_appearance
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from scipy.spatial.transform import Rotation

from synthetic_signwriting.hands.hands import load_hands_3d, get_hand_signwriting_symbol


class Keyframe(NamedTuple):
    face: np.ndarray

    left_hand: np.ndarray
    left_hand_signwriting: int

    right_hand: np.ndarray
    right_hand_signwriting: int


def unshift_hand(pose: Pose, hand_component: str):
    # TODO: move to pose library
    # pylint: disable=protected-access
    wrist_index = pose.header._get_point_index(hand_component, "WRIST")
    hand = pose.body.data[:, :, wrist_index: wrist_index + 21]
    body_wrist_name = "LEFT_WRIST" if hand_component == "LEFT_HAND_LANDMARKS" else "RIGHT_WRIST"
    # pylint: disable=protected-access
    body_wrist_index = pose.header._get_point_index("POSE_LANDMARKS", body_wrist_name)
    body_wrist = pose.body.data[:, :, body_wrist_index: body_wrist_index + 1]
    pose.body.data[:, :, wrist_index: wrist_index + 21] = hand + body_wrist


def get_corrected_mean_appearance():
    pose = get_mean_appearance()
    # pylint: disable=protected-access
    pose_right_elbow = pose.header._get_point_index("POSE_LANDMARKS", "RIGHT_ELBOW")
    # pylint: disable=protected-access
    pose_left_elbow = pose.header._get_point_index("POSE_LANDMARKS", "LEFT_ELBOW")
    pose.body.data[0, 0, pose_right_elbow] = pose.body.data[0, 0, pose_left_elbow] * np.array([-1, 1, 1])
    return pose


class SyntheticSignWritingGenerator:
    def __init__(self):
        self.keyframes = []
        self.arm_length = random.randint(550, 650)

    def _get_random_hand(self, is_right_hand=False, hand_index=None, hand_orientation=None, hand_plane=None):
        """
        Get a random hand array in 3D from the hands dataset.
        Randomly rotate it in 3D space around the X axis to simulate parallel to the wall/floor plane.
        Randomly rotate it in 3D space around the Y axis to simulate facing inwards/outwards/sideways.
        """
        hands = load_hands_3d()
        if hand_index is None:
            hand_index = random.randint(0, len(hands) - 1)
        hand = hands[hand_index]
        symbol = get_hand_signwriting_symbol(hand_index)

        # TODO: if our assumption is correct, and the hand is facing out
        symbol += 0x20

        # 33% chance to face hand inwards
        if hand_orientation is None:
            hand_orientation = random.random()
        if hand_orientation < 0.33:
            # hand faces inwards
            hand = Rotation.from_euler("y", 180, degrees=True).apply(hand)
            symbol -= 0x20
        elif hand_orientation < 0.66:
            # hand faces middle
            hand = Rotation.from_euler("y", -90, degrees=True).apply(hand)
            symbol -= 0x10

        # 50% chance to move hand to floor plane
        if hand_plane is None:
            hand_plane = random.random()
        if hand_plane > 0.5:
            hand = Rotation.from_euler("x", -90, degrees=True).apply(hand)
            symbol += 0x30

        # Mirror right hand
        if is_right_hand:
            hand = hand * np.array([-1, 1, 1])

        return hand, symbol

    def add_keyframe(self):
        right_hand, right_hand_signwriting = self._get_random_hand(is_right_hand=True)
        left_hand, left_hand_signwriting = self._get_random_hand(is_right_hand=False)

        keyframe = Keyframe(face=None,
                            left_hand=left_hand,
                            left_hand_signwriting=left_hand_signwriting,
                            right_hand=right_hand,
                            right_hand_signwriting=right_hand_signwriting)
        self.keyframes.append(keyframe)

    # pylint: disable=too-many-arguments, too-many-locals
    def _move_arm(self, pose: Pose, hand: str,
                  start_xy_angle: int, start_xz_angle: int,
                  end_xy_angle: int, end_xz_angle: int):
        pose_length = len(pose.body.data)
        # pylint: disable=protected-access
        hand_index = pose.header._get_point_index("POSE_LANDMARKS", f"{hand.upper()}_WRIST")
        # pylint: disable=protected-access
        elbow_index = pose.header._get_point_index("POSE_LANDMARKS", f"{hand.upper()}_ELBOW")

        x_sign = 1 if hand == "right" else -1

        for i in range(pose_length):
            x, y, z = pose.body.data[i, 0, elbow_index]
            xy_angle = start_xy_angle + (end_xy_angle - start_xy_angle) * i / pose_length
            xz_angle = start_xz_angle + (end_xz_angle - start_xz_angle) * i / pose_length

            x = x + self.arm_length * np.cos(np.radians(xy_angle)) * np.cos(np.radians(xz_angle)) * x_sign
            y = y + self.arm_length * np.sin(np.radians(xy_angle)) * np.cos(np.radians(xz_angle))
            z = z + self.arm_length * np.cos(np.radians(xy_angle)) * np.sin(np.radians(xz_angle))
            pose.body.data[i, 0, hand_index] = np.array([x, y, z])

    def create_hand_raise(self, pose: Pose, num_frames=None):
        if num_frames is None:
            num_frames = random.randint(5, 20)

        # create N frames (for start and end)
        pose_body = NumPyPoseBody(fps=25,
                                  data=np.repeat(pose.body.data, num_frames, axis=0),
                                  confidence=np.repeat(pose.body.confidence, num_frames, axis=0))
        raise_hands_pose = Pose(header=pose.header, body=pose_body)

        # Move arm
        end_xy_angle = -random.randint(60, 90)
        start_xy_angle = random.randint(50, 70)
        xz_angle = random.randint(20, 40)
        self._move_arm(raise_hands_pose, "right", start_xy_angle, xz_angle, end_xy_angle, xz_angle)
        self._move_arm(raise_hands_pose, "left", start_xy_angle, xz_angle, end_xy_angle, xz_angle)

        return raise_hands_pose

    def create_pose_segments(self):
        pose = get_corrected_mean_appearance()
        hand_raising = self.create_hand_raise(pose)

        # Start with hand raising
        pose_segments = [hand_raising]
        # Create segments for each transition
        for _ in range(len(self.keyframes) - 1):
            num_frames = random.randint(10, 15)
            last_frame_data = pose_segments[-1].body.data[-1]
            last_frame_confidence = pose_segments[-1].body.confidence[-1]
            segment_body = NumPyPoseBody(fps=25, data=np.stack([last_frame_data] * num_frames, axis=0),
                                         confidence=np.stack([last_frame_confidence] * num_frames, axis=0))
            segment_pose = Pose(header=pose.header, body=segment_body)
            pose_segments.append(segment_pose)
        # End with hand lowering (copy is necessary)
        hand_lowering_body = NumPyPoseBody(fps=25, data=hand_raising.body.data.copy()[::-1],
                                           confidence=hand_raising.body.confidence.copy()[::-1])
        hand_lowering = Pose(header=pose.header, body=hand_lowering_body)
        pose_segments.append(hand_lowering)

        return pose_segments

    def render_segment(self, pose: Pose, last_keyframe: Keyframe, keyframe: Keyframe):
        hand_points = 21

        for hand_name in ["right", "left"]:
            last_keyframe_hand = getattr(last_keyframe, f"{hand_name}_hand")
            hand = getattr(keyframe, f"{hand_name}_hand")
            hand_component = f"{hand_name.upper()}_HAND_LANDMARKS"
            if hand is not None:
                # pylint: disable=protected-access
                body_elbow_point = pose.header._get_point_index("POSE_LANDMARKS",
                                                                f"{hand_name.upper()}_ELBOW")
                # pylint: disable=protected-access
                body_wrist_point = pose.header._get_point_index("POSE_LANDMARKS",
                                                                f"{hand_name.upper()}_WRIST")

                for frame_index in range(len(pose.body.data)):
                    # naive implementation of hand morphing
                    frame_hand = last_keyframe_hand + (hand - last_keyframe_hand) * frame_index / len(pose.body.data)

                    # 3D rotate hand based on angle between elbow and wrist using scipy
                    elbow = pose.body.data[frame_index, 0, body_elbow_point]
                    wrist = pose.body.data[frame_index, 0, body_wrist_point]
                    y_z_angle = np.arctan2(elbow[2] - wrist[2], elbow[1] - wrist[1])
                    if hand_name == "right":
                        y_z_angle = -y_z_angle
                    rotation = Rotation.from_euler("xyz", [0, 0, y_z_angle])
                    rotated_hand = rotation.apply(frame_hand)

                    hand_index = pose.header._get_point_index(hand_component, "WRIST")
                    pose.body.data[frame_index, 0, hand_index:hand_index + hand_points] = rotated_hand
                unshift_hand(pose, hand_component)

    def render(self):
        segments = self.create_pose_segments()

        # By observation, find an open, "relaxed" hand
        relaxed_hand_index = random.randint(90, 96)
        relaxed_left_hand, _ = self._get_random_hand(is_right_hand=False, hand_index=relaxed_hand_index)
        relaxed_right_hand, _ = self._get_random_hand(is_right_hand=True, hand_index=relaxed_hand_index)

        # Last keyframe has two meanings: the final keyframe, and the first keyframe
        last_keyframe = Keyframe(face=None,
                                 left_hand=relaxed_left_hand,
                                 left_hand_signwriting=None,
                                 right_hand=relaxed_right_hand,
                                 right_hand_signwriting=None)
        last_keyframe = self.keyframes[0]
        # keyframes = self.keyframes + [last_keyframe]
        keyframes = self.keyframes + [last_keyframe]
        for segment, keyframe in zip(segments, keyframes):
            self.render_segment(segment, last_keyframe=last_keyframe, keyframe=keyframe)
            last_keyframe = keyframe

        # concatenate segments with couple of freeze frames in between
        body_data = []
        body_confidence = []
        for segment in segments:
            body_data.append(segment.body.data)
            body_data.append(np.stack([segment.body.data[-1]] * 3, axis=0))

            body_confidence.append(segment.body.confidence)
            body_confidence.append(np.stack([segment.body.confidence[-1]] * 3, axis=0))
        body = NumPyPoseBody(fps=25, data=np.concatenate(body_data, axis=0),
                             confidence=np.concatenate(body_confidence, axis=0))
        pose = Pose(header=segments[0].header, body=body)
        pose.focus()
        return pose


if __name__ == "__main__":
    synthetic = SyntheticSignWritingGenerator()
    synthetic.add_keyframe()
    synthetic.add_keyframe()
    generated_pose = synthetic.render()
    generated_pose = reduce_holistic(generated_pose)
    generated_pose.focus()

    with open("pose.pose", "wb") as f:
        generated_pose.write(f)

    # visualizer = PoseVisualizer(generated_pose)
    # visualizer.save_gif("pose.gif", tqdm(visualizer.draw(), unit="frame"))

    # I would like an example where I could fingerspell
    # add "A"
    # add "M"
    # add "I"
    # add "T"
