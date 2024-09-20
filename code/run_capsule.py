""" Runs capsule """

import os
from pathlib import Path

import numpy as np

from get_statistics import z1_multichannel_stats
from utils import utils


def run():
    """Runs large-scale statistics for z1 data"""

    # Code Ocean folders
    RESULTS_FOLDER = Path(os.path.abspath("../results"))
    # SCRATCH_FOLDER = Path(os.path.abspath("../scratch"))
    DATA_FOLDER = Path(os.path.abspath("../data"))

    # Output folder
    output_folder = RESULTS_FOLDER.joinpath("puncta_stats")
    utils.create_folder(dest_dir=str(output_folder), verbose=True)

    logger = utils.create_logger(output_log_path=str(output_folder))

    IMAGE_PATH = f"{DATA_FOLDER}/HCR_736207.01_2024-07-25_13-00-00"

    stats_parameters = {"buffer_radius": 6, "context_radius": 3, "bkg_percentile": 1}

    multichannel_spots = {
        "488": np.load(
            f"{DATA_FOLDER}/HCR_736207-01_2024-07-25_13-00-00-spots-488/spots.npy"
        ),
        "638": np.load(
            f"{DATA_FOLDER}/HCR_736207_01_2024-07-25_13-00-00-spots-638/spots.npy"
        ),
    }

    dataset_path = f"{IMAGE_PATH}/fused/channel_488.zarr"
    z1_multichannel_stats(
        dataset_path=dataset_path,
        multiscale="0",
        multichannel_spots=multichannel_spots,
        prediction_chunksize=(128, 128, 128),
        target_size_mb=3048,
        n_workers=0,
        axis_pad=14,
        batch_size=1,
        output_folder=output_folder.joinpath("channel_4"),
        stats_parameters=stats_parameters,
        logger=logger,
        super_chunksize=None,
    )


if __name__ == "__main__":
    run()
