""" Runs capsule """

import os
from pathlib import Path

from get_statistics import z1_multichannel_stats

from .utils import utils


def run():
    """Runs large-scale statistics for z1 data"""

    # Code Ocean folders
    RESULTS_FOLDER = Path(os.path.abspath("../results"))
    # SCRATCH_FOLDER = Path(os.path.abspath("../scratch"))
    DATA_FOLDER = Path(os.path.abspath("../data"))

    # Output folder
    output_folder = RESULTS_FOLDER.joinpath("puncta_detection_results")
    utils.create_folder(dest_dir=str(output_folder), verbose=True)

    logger = utils.create_logger(output_log_path=str(output_folder))

    IMAGE_PATH = "s3://aind-open-data/HCR_BL6-000_2023-06-1_00-00-00_fused_2024-04-02_20-06-14/channel_4.zarr"

    stats_parameters = {"buffer_radius": 6, "context_radius": 3, "bkg_percentile": 1}

    z1_multichannel_stats(
        dataset_path=IMAGE_PATH,
        multiscale="0",
        prediction_chunksize=(128, 128, 128),
        target_size_mb=3048,
        n_workers=0,
        axis_pad=14,
        batch_size=1,
        output_folder="./results",
        stats_parameters=stats_parameters,
        logger=logger,
        super_chunksize=None,
    )


if __name__ == "__main__":
    run()
