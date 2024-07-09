"""
For each of the ZYX identified spots in every channel,
we need to get spot statistics in the other channels.
This is helpful for spot demixing. E.g., a dataset with
4 channels where we have all the ZYX locations of spots.

    - spots_ch0.npy, spots_ch1.npy, spots_ch2.npy, spots_ch3.npy, 

For channel 0, we need to get statistics in all the ZYX locations
from ch1, ch2, and ch3.
"""

from typing import Iterable, List, Optional, Tuple

import numpy as np
from _shared.types import ArrayLike


def scan_bbox(
    img: ArrayLike, spots: ArrayLike, radius: int
) -> Iterable[Tuple[List, ArrayLike]]:
    """
    Scans the spots to get image data
    for each of them. Additionally, we prune
    spots around borders of the image since
    the algorithm is sensitive to these areas
    and we also have padding.

    Parameters
    ----------
    img: ArrayLike
        Image where the spots are located

    spots: ArrayLike
        Identified spots

    radius: int
        Search radius around each spot

    Returns
    -------
        Iterable with array with image data of shape radius in each axis.
    """
    depth, height, width = img.shape

    for p in spots:
        z_min = int(max(0, p[0] - radius))
        z_max = int(min(depth - 1, p[0] + radius))

        y_min = int(max(0, p[1] - radius))
        y_max = int(min(height - 1, p[1] + radius))

        x_min = int(max(0, p[2] - radius))
        x_max = int(min(width - 1, p[2] + radius))

        yield p, img[
            z_min:z_max,  # noqa: E203
            y_min:y_max,  # noqa: E203
            x_min:x_max,  # noqa: E203
        ]


def estimate_background_foreground(
    buffer_context: ArrayLike,
    context_radius: int,
    background_percentile: Optional[float] = 1.0,
) -> Tuple[float, float]:
    """
    Estimates the background foreground and background
    of a given spot. The spot must be in the center of
    the buffer context image.

    Parameters
    ----------
    buffer_context: ArrayLike
        Spot data around a buffer radius
        which should be bigger or equal to
        context radius. The spot must be in
        the center of this block of data.

    context_radius: int
        Radius used when fitting the gaussian.

    background_percentile: Optional[float]
        Background percentile to use.
        Default: 1.0

    Returns
    -------
    Tuple[float, float]
        Foreground and background spot intensities.
    """
    spot_center = np.array(buffer_context.shape) // 2

    # Grid for the spherical mask
    z, y, x = np.ogrid[
        : buffer_context.shape[0],
        : buffer_context.shape[1],
        : buffer_context.shape[2],
    ]

    # Condition to create a circular mask
    fg_condition = (
        (z - spot_center[0]) ** 2
        + (y - spot_center[1]) ** 2
        + (x - spot_center[2]) ** 2
    ) <= context_radius**2

    bg_condition = np.bitwise_not(fg_condition)

    # Background intensities
    bg_intensities = buffer_context * bg_condition
    bg_intensities = bg_intensities[bg_intensities != 0]

    # Foreground intensities
    fg_intensities = buffer_context * fg_condition
    fg_intensities = fg_intensities[fg_intensities != 0]

    # Manual inspection of spot
    spot_bg = -1
    spot_fg = -1

    if not bg_intensities.shape[0]:
        print(
            f"Problem in spot {spot_center}, non-zero background is: {bg_intensities}"
        )

    else:
        # Getting background percentile
        spot_bg = np.percentile(bg_intensities.flatten(), background_percentile)

    if not fg_intensities.shape[0]:
        print(
            f"Problem in spot {spot_center}, non-zero foreground is: {fg_intensities}"
        )

    else:
        # Getting foreground mean
        spot_fg = np.mean(fg_intensities)

    return spot_fg, spot_bg


def get_spot_chn_stats(
    data_block: ArrayLike,
    spots_in_block: ArrayLike,
    buffer_radius: int,
    context_radius: Optional[int] = 3,
    background_percentile: Optional[float] = 1.0,
):
    """
    Given a block of data and 3D points
    representing spots within this 3D block,
    we need to get spot statistics.

    Parameters
    ----------
    data_block: ArrayLike
        Block of data where we want to compute statistics
        from the spots in block.

    spots_in_block: ArrayLike
        Numpy array with the ZYX locations of the spots in
        the provided block of data. If running in a large
        dataset, make sure you shift the points to the
        local coordinate system of the data_block.

    buffer_radius: Int
        Buffer radius. This must be bigger than
        the context radius.

    context_radius: Optional[int] = 3
        Radius that was used by the spot detection
        algorithm when getting the area to compute
        the 3D gaussian fitting. Please, check
        https://github.com/AllenNeuralDynamics/aind-z1-spot-detection

    background_percentile: Optional[float]
        Background percentile used to compute
        the spot background.
        Default: 1.0
    """

    # Estimating spots foreground - background
    spots_fg_bg = np.array(
        [
            estimate_background_foreground(
                buffer_context=buffer_context,
                background_percentile=background_percentile,
                context_radius=context_radius,
            )
            for _, buffer_context in scan_bbox(
                data_block, spots_in_block, buffer_radius
            )
        ]
    )

    # horizontal stacking
    spots_in_block = np.append(spots_in_block.T, spots_fg_bg.T, axis=0).T

    return spots_in_block
