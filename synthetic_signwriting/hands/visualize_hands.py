import numpy as np
from pose_format.pose_visualizer import PoseVisualizer
from signwriting.visualizer.visualize import signwriting_to_image

from synthetic_signwriting.hands.hands import load_hands_3d, hands_to_pose, get_hand_signwriting

hands = load_hands_3d()
pose = hands_to_pose(hands)
pose.focus()


def draw_signwriting_on_frames(frames: np.ndarray, signwriting: list[str]):
    for frame, sign in zip(frames, signwriting):
        signwriting_img = signwriting_to_image(sign, trust_box=False)  # Pillow image
        signwriting_img_rgb = signwriting_img.convert("RGB")

        # Draw signwriting (pillow image) on the frame (np array)
        padding = 10
        frame[padding:padding + signwriting_img.height,
        padding:padding + signwriting_img.width] = np.array(signwriting_img_rgb)

        yield frame


visualizer = PoseVisualizer(pose)
modified_frames = draw_signwriting_on_frames(frames=visualizer.draw(),
                                             signwriting=[get_hand_signwriting(i) for i in range(len(pose.body.data))])
visualizer.save_gif("hand.gif", modified_frames)

print(get_hand_signwriting(1))
