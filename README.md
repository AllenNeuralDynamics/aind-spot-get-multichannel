# aind-spot-get-multichannel

Repository that computes statistics of ZYX spots in different channels. This is helpful for spot demixing. E.g., a dataset with 4 channels where we have all the ZYX locations of spots.
    - spots_ch0.npy, spots_ch1.npy, spots_ch2.npy, spots_ch3.npy.

For channel 0, we need to get statistics in all the ZYX locations from ch1, ch2, and ch3. Currently, the capsule generates the following outputs:

If we are computing stats for channel 488 with spots from channel 488 and 638, the output folder structure will look like:

- image_data_channel_488_versus_spots_488.csv - This means that the image data came from channel 488 and the spots used for foreground and background estimation were from channel 488.

- image_data_channel_488_versus_spots_638.csv - This means that the image data came from channel 488 and the spots used for foreground and background estimation were from channel 638.

The CSV has the following columns:

- Z: Z position of the spot.
- Y: Y position of the spot.
- X: X position of the spot.
- SEG_ID: Segmentation ID of the cell the spot belongs to.
- FG: Spot foreground intensity.
- BG: Spot background intensity.