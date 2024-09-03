#!/bin/bash

# Create a JSON file with the detector parameters
cat << EOF > detector.json
{
    "pixel_size_x": 0.075,
    "pixel_size_y": 0.075,
    "distance": 306.765,
    "beam_center_x": 1597.11,
    "beam_center_y": 1691.12
}
EOF

# Open file descriptor 3 for writing to output_file.txt
exec 3> output_file.txt

# Extended dispersion test dataset
./spotfinder /dls/i24/data/2024/nr27313-319/gw/Test_Insulin/ins_big_15/ins_big_15_2_master.h5 \
  --min-spot-size 3 \
  --pipe_fd 3 \
  --dmin 4 \
  --wavelength 0.619901 \
  --detector "$(cat detector.json)" \
  --algorithm "dispersion_extended" \
  --images 1 \
  --writeout

# Close file descriptor 3
exec 3>&-

# Delete the JSON file
rm detector.json