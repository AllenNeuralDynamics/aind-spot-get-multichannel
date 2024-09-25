# aind-spot-get-multichannel

Repository that computes statistics of ZYX spots in different channels. This is helpful for spot demixing. E.g., a dataset with 4 channels where we have all the ZYX locations of spots.
    - spots_ch0.npy, spots_ch1.npy, spots_ch2.npy, spots_ch3.npy.

For channel 0, we need to get statistics in all the ZYX locations from ch1, ch2, and ch3. Currently, the capsule generates the following outputs:

If we are computing stats for channel 488 with spots from channel 488 and 638, the output folder structure will look like:

- image_data_channel_488_versus_spots_488.csv - This means that the image data came from channel 488 and the spots used for foreground and background estimation were from channel 488.

- image_data_channel_488_versus_spots_638.csv - This means that the image data came from channel 488 and the spots used for foreground and background estimation were from channel 638.

The CSV has the following columns:

The output of this algorithm is a CSV file with the following columns. These columns come from the provided CSV except from foreground (FG) and background (BG):

- Z: Z location of the spot.
- Y: Y location of the spot.
- X: X location of the spot.
- Z_center: Z center of the spot during the guassian fitting, useful for demixing.
- Y_center: Y center of the spot during the guassian fitting, useful for demixing.
- X_center: X center of the spot during the guassian fitting, useful for demixing.
- dist: Euclidean distance or L2 norm of the ZYX center vector, $`norm = \sqrt{z^2 + y^2 + x^2}`$.
- r: Pearson correlation coefficient between integrated 3D gaussian and the 3D context where the spot is located.
- SEG_ID (optional): When a segmentation mask is provided, a new column is added with the segmentation ID of the detected spot.
- FG: Spot foreground intensity.
- BG: Spot background intensity.