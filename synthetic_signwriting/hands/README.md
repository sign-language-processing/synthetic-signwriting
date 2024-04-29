# Hands

To get the full list of hands in SignWriting we can use
https://github.com/sign-language-processing/3d-hands-benchmark

It has MediaPipe poses from various crops and rotations.
So in this part we have two options:

1. Stick with the 6 photographed angles, and have 6 variants of every hand shape
   (where perhaps the pose is the mean from different crops).
   This method is more realistic, as it uses real data directly.
2. Construct 3D poses, and consider the ground truth 3D pose to be the mean of the 6 views, across all crops.
   Then, in real time, rotate the 3D pose to the desired angle.
   This method looks better, and is easier for the NN, since the 3D poses are more accurate.