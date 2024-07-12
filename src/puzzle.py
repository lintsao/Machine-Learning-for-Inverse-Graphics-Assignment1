from pathlib import Path
from typing import Literal, TypedDict

from jaxtyping import Float
import torch
from torch import Tensor

import json
import os


class PuzzleDataset(TypedDict):
    extrinsics: Float[Tensor, "batch 4 4"]
    intrinsics: Float[Tensor, "batch 3 3"]
    images: Float[Tensor, "batch height width"]


def load_dataset(path: Path) -> PuzzleDataset:
    """Load the dataset into the required format."""

    # Load JSON file.
    with open(os.path.join(path, "metadata.json"), 'r') as file:
        metadata = json.load(file)
    # extrinsics = metadata["extrinsics"]
    # intrinsics = metadata["intrinsics"]

    return metadata

def normalize_vector(v):
    norm = torch.norm(v)
    if norm == 0: 
        return v
    return v / norm

def convert_to_opencv_format(extrinsic):
    # World to Cam
    # Extract rotation matrix and translation vector
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]

    # Compute look vector (from camera position to origin)
    camera_position_in_world = (torch.inverse(extrinsic) @ (Tensor([0, 0, 0, 1]).T))[:3]
    look_vector = normalize_vector(-camera_position_in_world)

    # Find the up vector (should be close to +y in world space)
    up_candidates = [R[:, i] for i in range(3)]
    dot_products = [torch.dot(up_candidate, torch.tensor([0, 1, 0], dtype=torch.float32)) for up_candidate in up_candidates]
    up_vector = normalize_vector(up_candidates[torch.argmax(torch.tensor(dot_products))])

    # Compute right vector using cross product
    right_vector = normalize_vector(torch.cross(up_vector, look_vector))

    # Recompute up vector to ensure orthogonality
    up_vector = torch.cross(look_vector, right_vector)

    # Construct the new rotation matrix in OpenCV format
    new_R = torch.stack([right_vector, -up_vector, look_vector], dim=1)

    # Construct the new extrinsic matrix
    new_extrinsic = torch.eye(4, dtype=torch.float32)
    new_extrinsic[:3, :3] = new_R
    new_extrinsic[:3, 3] = t

    return new_extrinsic


def convert_dataset(dataset: PuzzleDataset) -> PuzzleDataset:
    """Convert the dataset into OpenCV-style camera-to-world format. As a reminder, this
    format has the following specification:

    - The camera look vector is +Z.
    - The camera up vector is -Y.
    - The camera right vector is +X.
    - The extrinsics are in camera-to-world format, meaning that they transform points
      in camera space to points in world space.

    The original dataset:
    - The extrinsics are either in camera-to-world format or world-to-camera format.
    - The axes have been randomized, meaning that the camera look, up, and right vectors could be any of (+x, -x, +y, -y, +z, -z).

    The cameras are arranged as described below. Use this information to help you figure out your camera format:
    - The camera origins are always exactly 2 units from the origin.
    - The world up vector is +y, and all cameras have y >= 0.
    - All camera look vectors point directly at the origin.
    - All camera up vectors are pointed "up" in the world. In other words, the dot product between any camera up vector and +y is positive.
    Hint: How might one build a rotation matrix to convert between camera coordinate systems?
    """
    extrinsics = Tensor(dataset["extrinsics"])
    intrinsics = Tensor(dataset["intrinsics"])
    new_extrinsics = []

    for extrinsic in extrinsics:
        new_extrinsic = convert_to_opencv_format(extrinsic)
        new_extrinsics.append(torch.linalg.inv(new_extrinsic))

    new_extrinsics = torch.stack(new_extrinsics)

    convert_data = {"extrinsics": new_extrinsics, "intrinsics": intrinsics}

    return convert_data


def quiz_question_1() -> Literal["w2c", "c2w"]:
    """In what format was your puzzle dataset?"""

    raise NotImplementedError("This is your homework.")


def quiz_question_2() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera look vector?"""

    raise NotImplementedError("This is your homework.")


def quiz_question_3() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera up vector?"""

    raise NotImplementedError("This is your homework.")


def quiz_question_4() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera right vector?"""

    raise NotImplementedError("This is your homework.")


def explanation_of_problem_solving_process() -> str:
    """Please return a string (a few sentences) to describe how you solved the puzzle.
    We'll only grade you on whether you provide a descriptive answer, not on how you
    solved the puzzle (brute force, deduction, etc.).
    """

    raise NotImplementedError("This is your homework.")
