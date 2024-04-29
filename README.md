# synthetic-signwriting

A utility to generate synthetic SignWriting poses, for pretraining machine learning models.

Following the success of classifying real human poses from limited, synthetic SignWriting data from:

- Hands: https://www.youtube.com/watch?v=pCKRWSNIaNQ
- Faces: https://www.youtube.com/watch?v=bwCNLiksILU

We aim to generate synthetic SignWriting poses to pretrain models for downstream tasks.

There are multiple ways to do this, such as:

1. Stitching together poses from a database, and smoothing the transitions
   (requires a large amount of work to make the transitions).
2. Using a generative model to generate poses from scratch (requires a model, we don't have).
3. Using an existing sign language avatar, mapping between the avatar and SignWriting, and generating poses from the
   avatar (requires an avatar).
4. Creating pose key-frames from a database, and using a denosing autoencoder to generate poses in between (requires a
   model).

## Installation

```bash
pip install git+https://github.com/sign-language-processing/synthetic-signwriting
```

## How does this work?

basically, the program chooses a hand shape or two, and orientation/rotation etc, and generates a pose that has this
hand shape (from a database of 3D poses)  - it generates a pose for the full body, and just morphs the hand shape over
time. It then yields the pose and the SignWriting representing this sign (handshape change)
Then an addition can be to have two handed signs, symmetric or not, that have the same hand shape or not, and generate
SignWriting. Another addition can be controlling the face - adding a smile, adding a wink, adding an eyebrow raise etc.
It can start from neutral and move towards different features, or already start at a given feature such as raised
eyebrows and stay there.
another improvement could be the inclusion of movement paths between the handshape changes - the software can choose a
movement arrow, and follow its path (in the x, y, or z axis) to showcase different movement patterns.
it is important that the hand position is not always at the same place, but is placed around a predefined center with
some standard deviation (the mean and standard deviation are data driven). When moving the palm, it should also move the
elbow, but not the shoulder
finally, a virtual "camera" is produced to capture the pose sequence from a standard view (things closer to the camera
are larger etc). The camera also has some parameters such as position (which creates tiny difference in the angles of
projection) or slight rotation.
the synthetic signwriting genretor always yields an infinite number of pose sequences (in different lengths) and a
signwriting sequence of a single sign.

The class should allow generating key-frames, modifying them (adding a hand shape for example) and then "render" which
does the smoothing.