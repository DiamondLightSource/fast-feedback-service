#!/bin/bash

# This script tests the dispersion_extended algorithm on a test dataset
# It is meant to be run from the build folder: working_directory/build

# Ensure the output directory exists
mkdir -p _debug_output

# Change to the output directory
cd _debug_output

# Create a JSON file
cat << EOF > detector.json
{
    "pixel_size_x": 0.075,
    "pixel_size_y": 0.075,
    "distance": 150,
    "beam_center_x": 1008.13,
    "beam_center_y": 1066.0
}
EOF

# Open file descriptor 3 for writing to output_file.txt
exec 3> output_file.txt

# Run the program with file descriptor 3
../bin/spotfinder /dls/i03/data/2024/cm37235-2/xraycentring/TestInsulin/ins_14/ins_14_24.nxs \
  --min-spot-size 3 \
  --pipe_fd 3 \
  --dmin 4 \
  --detector "$(cat detector.json)" \
  --algorithm "dispersion" \
  --threads 1 \
  --images 1

# Close file descriptor 3
exec 3>&-

# Delete the JSON file
rm detector.json