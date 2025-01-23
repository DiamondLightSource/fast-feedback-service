#!/bin/bash

# This script tests the dispersion_extended algorithm on a test dataset
# It is meant to be run from the build folder: working_directory/build

# Ensure the output directory exists
mkdir -p _debug_output

# Change to the output directory
cd _debug_output

# Create a JSON file
# cat << EOF > detector.json
# {
#     "pixel_size_x": 0.075,
#     "pixel_size_y": 0.075,
#     "distance": 150,
#     "beam_center_x": 75.60975,
#     "beam_center_y": 79.95
# }
# EOF

# Open file descriptor 3 for writing to output_file.txt
exec 3> output_file.txt

../bin/spotfinder /dls/i03/data/2024/cm37235-2/TestInsulin/ins_14/ins_14_49_45_master.h5 \
  --min-spot-size 3 \
  --pipe_fd 3 \
  --dmin 4 \
  --algorithm "dispersion" \
  --threads 1 \
  --images 1

# Close file descriptor 3
exec 3>&-

# Delete the JSON file
# rm detector.json

# Read the output from the file
output=$(cat output_file.txt)

# Define the expected output
expected_output='{"file":"/dls/i03/data/2024/cm37235-2/TestInsulin/ins_14/ins_14_49_45_master.h5","file-number":0,"n_spots_total":83,"num_strong_pixels":452}'

# Compare the output with the expected output
if [ "$output" == "$expected_output" ]; then
  echo "#############################################"
  echo "#                                           #"
  echo "#               SUCCESS!!!                  #"
  echo "#                                           #"
  echo "#############################################"
else
  echo "*********************************************"
  echo "*                                           *"
  echo "*               FAILURE!!!                  *"
  echo "*                                           *"
  echo "*********************************************"
fi