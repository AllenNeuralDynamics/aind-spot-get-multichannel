"""
Large-scale computation of spot statistics
"""

import logging
import multiprocessing
import os
from pathlib import Path
# from functools import partial
from time import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
from aind_large_scale_prediction.generator.dataset import create_data_loader
from aind_large_scale_prediction.generator.utils import (
    recover_global_position, unpad_global_coords)
from aind_large_scale_prediction.io import ImageReaderFactory

from _shared.types import ArrayLike, PathLike
from get_spot_chn_stats import get_spot_chn_stats
from utils import utils


def remove_points_in_pad_area(
    points: ArrayLike, unpadded_slices: Tuple[slice]
) -> ArrayLike:
    """
    Removes points in padding area. The padding is provided
    by the scheduler as well as the unpadded slices which
    will be used to remove points in those areas.

    Parameters
    ----------
    points: ArrayLike
        3D points in the chunk of data. When masks are provided,
        points will be 4D with an extra dimension for the mask id
        which is not modified.

    unpadded_slices: Tuple[slice]
        Slices that point to the non-overlapping area between chunks
        of data.

    Returns
    -------
    ArrayLike
        Points within the non-overlapping area.
    """

    # Validating seeds are within block boundaries
    unpadded_points = points[
        (points[:, 0] >= unpadded_slices[0].start)  # within Z boundaries
        & (points[:, 0] < unpadded_slices[0].stop)
        & (points[:, 1] >= unpadded_slices[1].start)  # Within Y boundaries
        & (points[:, 1] < unpadded_slices[1].stop)
        & (points[:, 2] >= unpadded_slices[2].start)  # Within X boundaries
        & (points[:, 2] < unpadded_slices[2].stop)
    ]

    return unpadded_points


def get_points_in_boundaries(
    points: ArrayLike,
    location_slices: Tuple[ArrayLike],
    shift: Optional[bool] = True,
    unpadded_local_slice=None,
) -> ArrayLike:
    """
    Returns the points that fill within a given
    3D block.

    Parameters
    ----------
    points: ArrayLike
        Points in the whole dataset.

    location_slices: Tuple[ArrayLike[int]]
        Slices of the 3D block of data
        with respect of the global coordinate system.

    shift: Optional[bool]
        If we want to shift the points to the local
        coordinate system.
        Default: True

    Returns
    -------
    ArrayLike
        Array with the extracted points.
    """
    # These are the global slices
    start_slices = location_slices[0]
    stop_slices = location_slices[1]

    start_condition = np.all(points[:, :3] >= start_slices, axis=1)
    stop_condition = np.all(points[:, :3] < stop_slices, axis=1)

    extracted_rows_global = points[np.where(start_condition & stop_condition)]

    if not extracted_rows_global.shape[0]:
        return None

    returned_spots = extracted_rows_global

    # If shifting is enabled, shift points to the local coordinate system
    if shift:
        extracted_rows_global[:, :3] = extracted_rows_global[:, :3] - start_slices
        returned_spots = remove_points_in_pad_area(
            points=extracted_rows_global, unpadded_slices=unpadded_local_slice
        )

    if not returned_spots.shape[0]:
        return None

    return returned_spots


def execute_worker(
    data_block: ArrayLike,
    multichannel_spots: ArrayLike,
    stats_parameters: Dict,
    batch_super_chunk: Tuple[slice],
    batch_internal_slice: Tuple[slice],
    overlap_prediction_chunksize: Tuple[int],
    dataset_shape: Tuple[int],
    logger: logging.Logger,
) -> np.array:
    """
    Function that executes each worker. It takes
    the combined gradients and follows the flows.

    Parameters
    ----------
    data_block: ArrayLike
        Block of data to process.

    multichannel_spots: Dict
        multichannel_spots: Dict[ArrayLike]
        Dictionary with the spots that we want to use to
        estimate the statistics from other channels.
        Example: {"channel_0": ArrayLike, ..., "channel_n": ArrayLike}

    stats_parameters: Dict
        Dictionary with the stats parameters.

    batch_super_chunk: Tuple[slice]
        Slices of the super chunk loaded in shared memory.

    batch_internal_slice: Tuple[slice]
        Internal slice of the current chunk of data. This
        is a local coordinate system based on the super chunk.

    overlap_prediction_chunksize: Tuple[int]
        Overlap between chunks

    dataset_shape: Tuple[int]
        Large-scale dataset shape.

    logger: logging.Logger
        Logging object

    Returns
    -------
    Dict
        Dictionary with the spot statistics per channel.
    """
    curr_pid = os.getpid()

    # Getting global coordinate system
    (
        global_coord_pos,
        global_coord_positions_start,
        global_coord_positions_end,
    ) = recover_global_position(
        super_chunk_slice=batch_super_chunk,
        internal_slices=batch_internal_slice,
    )

    # Unpadding global coordinate system
    unpadded_global_slice, unpadded_local_slice = unpad_global_coords(
        global_coord_pos=global_coord_pos[-3:],
        block_shape=data_block.shape[-3:],
        overlap_prediction_chunksize=overlap_prediction_chunksize[-3:],
        dataset_shape=dataset_shape[-3:],  # zarr_dataset.lazy_data.shape,
    )

    #     print(f"Worker {os.getpid()} -> internal slice: {batch_internal_slice} - superchunk: {batch_super_chunk} - Global pos: {global_coord_pos} - unpadded: {unpadded_global_slice} - unpadded local: {unpadded_local_slice}")
    #     exit()
    worker_spt_chn_statistics = {}

    for spot_channel_name, global_spots in multichannel_spots.items():

        # Getting spots in the block and shifting them to local coord
        spots_in_block = get_points_in_boundaries(
            points=global_spots,
            location_slices=(
                np.array(global_coord_positions_start),
                np.array(global_coord_positions_end),
            ),
            shift=True,
            unpadded_local_slice=unpadded_local_slice,
        )

        # If no spots in block, continue
        if spots_in_block is None:
            worker_spt_chn_statistics[spot_channel_name] = None
            continue

        # Getting spots with statistics
        spot_with_statistics = get_spot_chn_stats(
            data_block=data_block,
            spots_in_block=spots_in_block,
            buffer_radius=stats_parameters["buffer_radius"],
            context_radius=stats_parameters["context_radius"],
            background_percentile=stats_parameters["bkg_percentile"],
        )

        # Recovering global coordinate system
        spot_with_statistics[:, :3] = np.array(global_coord_positions_start)[
            :, -3:
        ] + np.array(spot_with_statistics[:, :3])

        logger.info(
            f"Worker {curr_pid}: Found {spot_with_statistics.shape} spots in global coords: {unpadded_global_slice}"
        )

        worker_spt_chn_statistics[spot_channel_name] = spot_with_statistics.copy()

    return worker_spt_chn_statistics


def _execute_worker(params: Dict):
    """
    Worker interface to provide parameters

    Parameters
    ----------
    params: Dict
        Dictionary with the parameters to provide
        to the execution function.
    """
    return execute_worker(**params)


def producer(
    producer_queue,
    zarr_data_loader,
    logger,
    n_consumers,
):
    """
    Function that sends blocks of data to
    the queue to be acquired by the workers.

    Parameters
    ----------
    producer_queue: multiprocessing.Queue
        Multiprocessing queue where blocks
        are sent to be acquired by workers.

    zarr_data_loader: DataLoader
        Zarr data loader

    logger: logging.Logger
        Logging object

    n_consumers: int
        Number of consumers
    """
    # total_samples = sum(zarr_dataset.internal_slice_sum)
    worker_pid = os.getpid()

    logger.info(f"Starting producer queue: {worker_pid}")
    for i, sample in enumerate(zarr_data_loader):

        producer_queue.put(
            {
                "data_block": sample.batch_tensor.numpy()[0, ...],
                "i": i,
                "batch_super_chunk": sample.batch_super_chunk[0],
                "batch_internal_slice": sample.batch_internal_slice,
            },
            block=True,
        )
        logger.info(f"[+] Worker {worker_pid} setting block {i}")

    for i in range(n_consumers):
        producer_queue.put(None, block=True)

    # zarr_dataset.lazy_data.shape
    logger.info(f"[+] Worker {worker_pid} -> Producer finished producing data.")


def consumer(
    queue,
    zarr_dataset,
    worker_params,
    results_dict,
):
    """
    Function executed in every worker
    to acquire data.

    Parameters
    ----------
    queue: multiprocessing.Queue
        Multiprocessing queue where blocks
        are sent to be acquired by workers.

    zarr_dataset: ArrayLike
        Zarr dataset

    worker_params: dict
        Worker parametes to execute a function.

    results_dict: multiprocessing.Dict
        Results dictionary where outputs
        are stored.
    """
    logger = worker_params["logger"]
    worker_results = {}
    worker_pid = os.getpid()
    logger.info(f"Starting consumer worker -> {worker_pid}")

    # Setting initial wait so all processes could be created
    # And producer can start generating data
    # sleep(60)

    # Start processing
    total_samples = sum(zarr_dataset.internal_slice_sum)

    while True:
        streamed_dict = queue.get(block=True)

        if streamed_dict is None:
            logger.info(f"[-] Worker {worker_pid} -> Turn off signal received...")
            break

        logger.info(
            f"[-] Worker {worker_pid} -> Consuming {streamed_dict['i']} - {streamed_dict['data_block'].shape} - Super chunk val: {zarr_dataset.curr_super_chunk_pos.value} - internal slice sum: {total_samples}"
        )

        # Simulate some processing
        worker_response = execute_worker(
            data_block=streamed_dict["data_block"],
            multichannel_spots=worker_params["multichannel_spots"],
            stats_parameters=worker_params["stats_parameters"],
            batch_super_chunk=streamed_dict["batch_super_chunk"],
            batch_internal_slice=streamed_dict["batch_internal_slice"],
            overlap_prediction_chunksize=worker_params["overlap_prediction_chunksize"],
            dataset_shape=worker_params["dataset_shape"],
            logger=logger,
        )

        for curr_channel_name, worker_spots in worker_response.items():

            if curr_channel_name not in worker_results:
                worker_results[curr_channel_name] = None

            if worker_spots is not None:
                worker_spots = worker_spots.astype(np.float32)

                # Adding worker response to multichannel global dictionary
                if worker_results[curr_channel_name] is None:
                    worker_results[curr_channel_name] = worker_spots.copy()

                else:
                    worker_results[curr_channel_name] = np.append(
                        worker_results[curr_channel_name],
                        worker_spots,
                        axis=0,
                    )

    logger.info(f"[-] Worker {worker_pid} -> Consumer finished consuming data.")
    results_dict[worker_pid] = worker_results


def z1_multichannel_stats(
    dataset_path: PathLike,
    multiscale: str,
    multichannel_spots: Dict,
    prediction_chunksize: Tuple[int, ...],
    target_size_mb: int,
    n_workers: int,
    axis_pad: int,
    batch_size: int,
    output_folder: str,
    stats_parameters: Dict,
    logger: logging.Logger,
    super_chunksize: Optional[Tuple[int, ...]] = None,
):
    """
    Chunked large-scale estimation of parameters
    of identified spots.

    Parameters
    ----------
    dataset_path: PathLike
        Path where the zarr dataset is stored. It could
        be a local path or in a S3 path.

    multiscale: str
        Multiscale to process

    multichannel_spots: Dict[ArrayLike]
        Dictionary with the spots that we want to use to
        estimate the statistics from other channels.
        Example: {"channel_0": ArrayLike, ..., "channel_n": ArrayLike}

    prediction_chunksize: Tuple[int, ...]
        Prediction chunksize the model will pull from
        the raw data

    target_size_mb: int
        Target size in megabytes the data loader will
        load in memory at a time

    n_workers: int
        Number of workers that will concurrently pull
        data from the shared super chunk in memory

    axis_pad: int
        Padding in each axis useful for the non-linear
        filtering.

    batch_size: int
        Batch size processed each time

    output_folder: str
        Output folder for the detected spots

    stats_parameters: Dict
        Dictionary with the stats parameters.

    logger: logging.Logger
        Logging object

    super_chunksize: Optional[Tuple[int, ...]]
        Super chunk size that will be in memory at a
        time from the raw data. If provided, then
        target_size_mb is ignored. Default: None

    """
    co_cpus = int(utils.get_code_ocean_cpu_limit())
    channel_name = Path(dataset_path).stem
    utils.create_folder(dest_dir=str(output_folder), verbose=True)

    if n_workers > co_cpus:
        raise ValueError(f"Provided workers {n_workers} > current workers {co_cpus}")

    logger.info(f"{20*'='} Running puncta detection {20*'='}")
    logger.info(f"Output folder: {output_folder}")

    utils.print_system_information(logger)

    logger.info(f"Processing dataset {dataset_path} with mulsticale {multiscale}")

    # Tracking compute resources
    # Subprocess to track used resources
    manager = multiprocessing.Manager()
    time_points = manager.list()
    cpu_percentages = manager.list()
    memory_usages = manager.list()

    profile_process = multiprocessing.Process(
        target=utils.profile_resources,
        args=(
            time_points,
            cpu_percentages,
            memory_usages,
            20,
        ),
    )
    profile_process.daemon = True
    profile_process.start()

    logger.info("Creating chunked data loader")
    shm_memory = psutil.virtual_memory()
    logger.info(f"Shared memory information: {shm_memory}")

    device = None

    pin_memory = True
    if device is not None:
        pin_memory = False
        multiprocessing.set_start_method("spawn", force=True)

    image_reader = ImageReaderFactory().create(
        data_path=dataset_path, parse_path=False, multiscale=multiscale
    )

    lazy_data = image_reader.as_dask_array()

    # image_metadata = image_reader.metadata()

    # logger.info(f"Full image metadata: {image_metadata}")

    # image_metadata = utils.parse_zarr_metadata(
    #     metadata=image_metadata, multiscale=multiscale
    # )

    # logger.info(f"Filtered Image metadata: {image_metadata}")

    overlap_prediction_chunksize = (axis_pad, axis_pad, axis_pad)
    zarr_data_loader, zarr_dataset = create_data_loader(
        lazy_data=lazy_data,
        target_size_mb=target_size_mb,
        prediction_chunksize=prediction_chunksize,
        overlap_prediction_chunksize=overlap_prediction_chunksize,
        n_workers=n_workers,
        batch_size=batch_size,
        dtype=np.float32,  # Allowed data type to process with pytorch cuda
        super_chunksize=super_chunksize,
        lazy_callback_fn=None,  # partial_lazy_deskewing,
        logger=logger,
        device=device,
        pin_memory=pin_memory,
        override_suggested_cpus=False,
        drop_last=True,
        locked_array=False,
    )

    logger.info(
        f"Running statistics. Prediction chunksize: {prediction_chunksize} - Overlap chunksize: {overlap_prediction_chunksize}"
    )

    start_time = time()

    total_batches = sum(zarr_dataset.internal_slice_sum) / batch_size

    # samples_per_iter = n_workers * batch_size
    logger.info(f"Number of batches: {total_batches}")

    # Variables for multiprocessing
    multichannel_final_spots = {key: None for key in list(multichannel_spots.keys())}

    # Setting exec workers to CO CPUs
    exec_n_workers = co_cpus

    # Create consumer processes
    factor = 10

    # Create a multiprocessing queue
    producer_queue = multiprocessing.Queue(maxsize=exec_n_workers * factor)

    worker_params = {
        "multichannel_spots": multichannel_spots,
        "stats_parameters": stats_parameters,
        "overlap_prediction_chunksize": overlap_prediction_chunksize,
        "dataset_shape": zarr_dataset.lazy_data.shape,
        "multichannel_final_spots": multichannel_final_spots,
        "logger": logger,
    }

    results_dict = manager.dict()

    logger.info(f"Setting up {exec_n_workers} workers...")
    consumers = [
        multiprocessing.Process(
            target=consumer,
            args=(
                producer_queue,
                zarr_dataset,
                worker_params,
                results_dict,
            ),
        )
        for _ in range(exec_n_workers)
    ]

    # Start consumer processes
    for consumer_process in consumers:
        consumer_process.start()

    # Main process acts as the producer
    producer(producer_queue, zarr_data_loader, logger, exec_n_workers)

    # Wait for consumer processes to finish
    for consumer_process in consumers:
        consumer_process.join()

    multichannel_final_spots = {key: None for key in list(multichannel_spots.keys())}
    spot_channel_names = list(multichannel_final_spots.keys())
    for worker_id, spots_channel in results_dict.items():

        for spot_channel_name in spot_channel_names:
            curr_spots = spots_channel[spot_channel_name]

            if curr_spots is None:
                continue

            logger.info(
                f"Worker {worker_id} computed {curr_spots.shape[0]} spots in channel {spot_channel_name}"
            )

            if multichannel_final_spots[spot_channel_name] is None:
                multichannel_final_spots[spot_channel_name] = curr_spots

            else:
                multichannel_final_spots[spot_channel_name] = np.append(
                    multichannel_final_spots[spot_channel_name],
                    curr_spots,
                    axis=0,
                )

    # Saving
    for spot_channel_name in multichannel_final_spots.keys():
        final_spots = multichannel_final_spots[spot_channel_name].astype(np.float32)
        print(f"Final spots channel {spot_channel_name} - {final_spots.shape}")

        spot_ch_df = pd.DataFrame(
            final_spots, columns=["Z", "Y", "X", "SEG_ID", "FG", "BG"]
        )

        spot_ch_df[["Z", "Y", "X"]] = spot_ch_df[["Z", "Y", "X"]].astype("int")
        spot_ch_df = spot_ch_df.sort_values(by="SEG_ID")

        # Saving spots
        spot_ch_df.to_csv(
            f"{output_folder}/image_data_{channel_name}_versus_spots_{spot_channel_name}.csv",
            index=False,
        )

    end_time = time()

    logger.info(f"Processing time: {end_time - start_time} seconds")

    # Getting tracked resources and plotting image
    utils.stop_child_process(profile_process)

    if len(time_points):
        utils.generate_resources_graphs(
            time_points,
            cpu_percentages,
            memory_usages,
            output_folder,
            "z1_stats",
        )


def main():
    """
    Main function to test the method
    """
    dataset_path = "s3://aind-open-data/HCR_BL6-000_2023-06-1_00-00-00_fused_2024-04-02_20-06-14/channel_1.zarr"
    # Code Ocean folders
    RESULTS_FOLDER = Path(os.path.abspath("../results"))

    # Output folder
    output_folder = RESULTS_FOLDER
    utils.create_folder(dest_dir=str(output_folder), verbose=True)
    logger = utils.create_logger(output_log_path=str(output_folder))
    stats_parameters = {"buffer_radius": 6, "context_radius": 3, "bkg_percentile": 1}
    multichannel_spots = {
        "channel_1": np.load("/Users/camilo.laiton/Downloads/spots_ch1.npy"),
    }

    z1_multichannel_stats(
        dataset_path=dataset_path,
        multiscale="0",
        multichannel_spots=multichannel_spots,
        prediction_chunksize=(128, 128, 128),
        target_size_mb=3048,
        n_workers=0,
        axis_pad=14,
        batch_size=1,
        output_folder=output_folder.joinpath("channel_1"),
        stats_parameters=stats_parameters,
        logger=logger,
        super_chunksize=None,
    )


if __name__ == "__main__":
    main()
