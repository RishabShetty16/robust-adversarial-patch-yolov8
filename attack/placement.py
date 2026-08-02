"""
placement.py

Patch placement utilities.

Responsibilities
----------------
- Compute patch placement coordinates.
- Support multiple placement modes.
- Ensure the patch stays inside image boundaries.

Author:
    Rishab Shetty
"""

import random


def get_patch_position(
    image_height,
    image_width,
    patch_size,
    placement_cfg,
    person_box=None,
):
    """
    Compute the (x, y) position for placing the patch.

    Parameters
    ----------
    image_height : int
    image_width : int
    patch_size : int
    placement_cfg : dict

    Returns
    -------
    (x, y)
    """

    mode = placement_cfg.get("mode", "fixed")

    # -------------------------------------------------
    # Fixed Placement
    # -------------------------------------------------

    if mode == "fixed":

        x = placement_cfg["x"]
        y = placement_cfg["y"]

    # -------------------------------------------------
    # Random Placement
    # -------------------------------------------------

    elif mode == "random":

        x = random.randint(
            0,
            max(0, image_width - patch_size),
        )

        y = random.randint(
            0,
            max(0, image_height - patch_size),
        )

    # -------------------------------------------------
    # Center Placement
    # -------------------------------------------------

    elif mode == "center":

        x = (image_width - patch_size) // 2

        y = (image_height - patch_size) // 2

    # -------------------------------------------------
    # Person Placement
    # -------------------------------------------------

    elif mode == "person":

        if person_box is None:

            # Fallback to center placement
            x = (image_width - patch_size) // 2
            y = (image_height - patch_size) // 2

        else:

            x1, y1, x2, y2 = person_box

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            x = center_x - patch_size // 2
            y = center_y - patch_size // 2

    # -------------------------------------------------
    # Unsupported Mode
    # -------------------------------------------------

    else:

        raise ValueError(
            f"Unknown placement mode: {mode}"
        )

    # -------------------------------------------------
    # Clamp
    # -------------------------------------------------

    x = max(
        0,
        min(
            x,
            image_width - patch_size,
        ),
    )

    y = max(
        0,
        min(
            y,
            image_height - patch_size,
        ),
    )

    return x, y