""" Runs capsule """

import os
from pathlib import Path

import numpy as np
import pandas as pd

from _shared.types import ArrayLike, PathLike
from get_statistics import z1_multichannel_stats
from utils import utils


def load_data(path: PathLike) -> ArrayLike:
    """
    Loads the spot data from a CSV or numpy array.

    Parameters
    ----------
    path: PathLike
        Path where the spots are stored.

    Raises
    ------
    ValueError
        If a format different than .csv or .npy
        is provided.

    Returns
    -------
    ArrayLike
        Numpy array with the spots.
    """

    path = Path(path)
    suffix = path.suffix
    data_np_format = None

    if suffix == ".csv":
        data_np_format = pd.read_csv(path).to_numpy().astype(np.float32)

    elif suffix == ".npy":
        data_np_format = np.load(path)

    else:
        raise ValueError(f"Only .npy and .csv are allowed. Received {suffix}")

    print(data_np_format.shape, data_np_format[0], data_np_format.dtype)
    return data_np_format


def run():
    """Runs large-scale statistics for z1 data"""

    # Code Ocean folders
    RESULTS_FOLDER = Path(os.path.abspath("../results"))
    # SCRATCH_FOLDER = Path(os.path.abspath("../scratch"))
    DATA_FOLDER = Path(os.path.abspath("../data"))

    # Output folder
    output_folder = RESULTS_FOLDER
    utils.create_folder(dest_dir=str(output_folder), verbose=True)

    logger = utils.create_logger(output_log_path=str(output_folder))

    IMAGE_PATH = f"{DATA_FOLDER}/HCR_736207.01_2024-07-25_13-00-00"

    stats_parameters = {"buffer_radius": 6, "context_radius": 3, "bkg_percentile": 1}

    multichannel_spots = {
        "488": load_data(
            f"{DATA_FOLDER}/HCR_736207-01_2024-07-25_13-00-00-spots-488/spots.npy"
        ),
        "638": load_data(
            f"{DATA_FOLDER}/HCR_736207_01_2024-07-25_13-00-00-spots-638/spots.npy"
        ),
    }

    image_data_channel = "channel_488"
    dataset_path = f"{IMAGE_PATH}/fused/{image_data_channel}.zarr"

    z1_multichannel_stats(
        dataset_path=dataset_path,
        multiscale="0",
        multichannel_spots=multichannel_spots,
        prediction_chunksize=(128, 128, 128),
        target_size_mb=3048,
        n_workers=0,
        axis_pad=14,
        batch_size=1,
        output_folder=output_folder.joinpath(image_data_channel),
        stats_parameters=stats_parameters,
        logger=logger,
        super_chunksize=None,
        segmentation_column=True,
    )


if __name__ == "__main__":
    run()
