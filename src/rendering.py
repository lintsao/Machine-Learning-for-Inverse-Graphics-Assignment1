from jaxtyping import Float
import torch
from torch import Tensor


def render_point_cloud(
    vertices: Float[Tensor, "vertex 3"],
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    resolution: tuple[int, int] = (256, 256),
) -> Float[Tensor, "batch height width"]:
    """Create a white canvas with the specified resolution. Then, transform the points
    into camera space, project them onto the image plane, and color the corresponding
    pixels on the canvas black.
    """
    # Image = Projection Matrix @ Extrinsic @ World Coordinates
    batch_size = extrinsics.shape[0]
    height, width = resolution
    device = vertices.device
    
    # Create a white canvas for each batch
    canvases = torch.ones(batch_size, height, width, dtype=torch.float32, device=device)

    # Convert vertices to homogeneous coordinates
    vertices_homogeneous = torch.cat([vertices, torch.ones(vertices.shape[0], 1)], dim=1)

    for i in range(batch_size):
        # Transform points into camera space using extrinsics
        camera_space_points = extrinsics[i] @ vertices_homogeneous.T  # [4, vertex]
        camera_space_points = camera_space_points.T  # [vertex, 4]

        # Project points onto the image plane using intrinsics
        projected_points = intrinsics[i] @ camera_space_points[:, :3].T  # [3, vertex]
        projected_points = projected_points.T  # [vertex, 3]

        # Normalize homogeneous coordinates
        projected_points = projected_points / projected_points[:, 2:3]  # [vertex, 3]

        # Convert to pixel coordinates
        pixel_coords = (projected_points[:, :2] * width).long()  # [vertex, 2]

        # Filter points that fall within the canvas bounds
        valid_mask = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < width) & (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < height)
        valid_pixel_coords = pixel_coords[valid_mask]

        # Color the corresponding pixels on the canvas black
        canvases[i, valid_pixel_coords[:, 1], valid_pixel_coords[:, 0]] = 0.0

    return canvases