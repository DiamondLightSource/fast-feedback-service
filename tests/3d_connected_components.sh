#!/bin/bash

# This script tests the 3d connected component algorithm on a test dataset
# It is meant to be run from the build folder: working_directory/build

# Ensure the output directory exists
mkdir -p _debug_output

# Change to the output directory
cd _debug_output

# Save the current value of the environment variable if it exists
if [ -z "${LOG_LEVEL+x}" ]; then
  original_value_set=false
else
  original_value_set=true
  original_value=$LOG_LEVEL
fi

# Change the log level to debug
export LOG_LEVEL="debug"

# Open file descriptor 3 for writing to output_file.txt
exec 3> output_file.txt

# ./bin/spotfinder /dls/i03/data/2024/cm37235-2/TestInsulin/ins_14/ins_14_49_45_master.h5 --min-spot-size 3 --dmin 4 --algorithm "dispersion" --threads 1 --images 5
../bin/spotfinder /dls/i03/data/2024/cm37235-2/TestInsulin/ins_14/ins_14_49_45_master.h5 \
  --min-spot-size 3 \
  --pipe_fd 3 \
  --dmin 4 \
  --algorithm "dispersion" \
  --threads 1 \
  --images 5

# Close file descriptor 3
exec 3>&-

# Restore the original value of the environment variable if it was set
if [ "$original_value_set" = true ]; then
  export LOG_LEVEL=$original_value
else
  unset LOG_LEVEL
fi

# Read the output from the file
# output=$(cat output_file.txt)

# Define the expected output
# expected_output='{ }'

# Compare the output with the expected output
# if [ "$output" == "$expected_output" ]; then
#   echo "#############################################"
#   echo "#                                           #"
#   echo "#               SUCCESS!!!                  #"
#   echo "#                                           #"
#   echo "#############################################"
# else
#   echo "*********************************************"
#   echo "*                                           *"
#   echo "*               FAILURE!!!                  *"
#   echo "*                                           *"
#   echo "*********************************************"
# fi