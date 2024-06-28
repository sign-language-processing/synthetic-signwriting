# Hands

To get the full list of hands in SignWriting we can use
https://github.com/sign-language-processing/3d-hands-benchmark

It has MediaPipe poses from various crops and rotations.
We construct 3D poses, and consider the ground truth 3D pose to be the median of the first 3 views, across 16 crops.
Then, in real time, we rotate the 3D pose to the desired angle.

```bash
wget https://github.com/sign-language-processing/3d-hands-benchmark/raw/master/benchmark/systems/mediapipe/v0.10.3.npy
```

![Hands](hands.gif)