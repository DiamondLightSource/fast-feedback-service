#!/bin/bash

# This script tests the dispersion_extended algorithm on a test dataset
# It is meant to be run from the build folder: working_directory/build

# Ensure the output directory exists
mkdir -p _debug_output

# Change to the output directory
cd _debug_output

# Create a JSON file with the detector parameters
# cat << EOF > detector.json
# {
#     "pixel_size_x": 0.075,
#     "pixel_size_y": 0.075,
#     "distance": 306.765,
#     "beam_center_x": 119.78358,
#     "beam_center_y": 126.83430
# }
# EOF

# Open file descriptor 3 for writing to output_file.txt
exec 3> output_file.txt

# Extended dispersion test dataset
# This is a good dataset to test the dispersion_extended algorithm. However this is 32bit data and requires truncation in order to be read
# ../bin/spotfinder /dls/i24/data/2024/nr27313-319/gw/Test_Insulin/ins_big_15/ins_big_15_2_master.h5 \
../bin/spotfinder /dls/i03/data/2024/cm37235-2/xraycentring/TestInsulin/ins_14/ins_14_24.nxs \
  --min-spot-size 3 \
  --pipe_fd 3 \
  --dmin 4 \
  --algorithm "dispersion_extended" \
  --images 1 \
  --writeout

# Close file descriptor 3
exec 3>&-

# Delete the JSON file
# rm detector.json

# Read the output from the file
output=$(cat output_file.txt)

# Define the expected output
expected_output='{"file":"/dls/i03/data/2024/cm37235-2/xraycentring/TestInsulin/ins_14/ins_14_24.nxs","file-number":0,"n_spots_total":3,"num_strong_pixels":115}'

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