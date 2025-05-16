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
  # Change the log level to debug
  export LOG_LEVEL="debug"
else
  original_value_set=true
fi

# Open file descriptor 3 for writing to output_file.txt
exec 3> output_file.txt

# ./bin/spotfinder /dls/i03/data/2024/cm37235-2/TestInsulin/ins_14/ins_14_49_45_master.h5 --min-spot-size 3 --dmin 4 --algorithm "dispersion" --threads 1 --images 5
../bin/spotfinder /dls/i03/data/2024/cm37235-2/TestInsulin/ins_14/ins_14_49_45_master.h5 \
  --min-spot-size 3 \
  --min-spot-size-3d 15 \
  --pipe_fd 3 \
  --dmin 4 \
  --algorithm "dispersion" \
  --threads 1 \
  --images 5 \
  --save-h5 \
  --writeout

# Close file descriptor 3
exec 3>&-

# Restore the original value of the environment variable if it was set
if [ "$original_value_set" = false ]; then
  unset LOG_LEVEL
fi

# Read the output from the files
output_file_content=$(head output_file.txt)
reflections_file_content=$(head 3d_reflections.txt)

# Define the expected output
expected_output_file='{"file":"/dls/i03/data/2024/cm37235-2/TestInsulin/ins_14/ins_14_49_45_master.h5","file-number":0,"n_spots_total":83,"num_strong_pixels":452}
{"file":"/dls/i03/data/2024/cm37235-2/TestInsulin/ins_14/ins_14_49_45_master.h5","file-number":1,"n_spots_total":82,"num_strong_pixels":442}
{"file":"/dls/i03/data/2024/cm37235-2/TestInsulin/ins_14/ins_14_49_45_master.h5","file-number":2,"n_spots_total":101,"num_strong_pixels":533}
{"file":"/dls/i03/data/2024/cm37235-2/TestInsulin/ins_14/ins_14_49_45_master.h5","file-number":3,"n_spots_total":102,"num_strong_pixels":559}
{"file":"/dls/i03/data/2024/cm37235-2/TestInsulin/ins_14/ins_14_49_45_master.h5","file-number":4,"n_spots_total":100,"num_strong_pixels":532}'

expected_reflections_file='X: [1896, 1899] Y: [3012, 3015] Z: [2, 4] COM: (1897.64, 3014.76, 3.3673)
X: [1923, 1926] Y: [2905, 2907] Z: [1, 4] COM: (1925.26, 2906.35, 2.92683)
X: [1388, 1390] Y: [2599, 2601] Z: [0, 4] COM: (1389.43, 2600.66, 2)
X: [1373, 1375] Y: [2576, 2579] Z: [1, 4] COM: (1374.19, 2577.93, 3.53085)
X: [2548, 2551] Y: [2357, 2360] Z: [0, 4] COM: (2550.05, 2359.19, 2.14211)
X: [2546, 2548] Y: [2341, 2343] Z: [0, 4] COM: (2547.98, 2343.23, 2.00769)
X: [2436, 2438] Y: [2220, 2221] Z: [0, 4] COM: (2437.74, 2220.86, 1.55614)
X: [1277, 1280] Y: [2209, 2213] Z: [0, 4] COM: (1279.14, 2211.93, 3.0243)
X: [1221, 1223] Y: [2202, 2204] Z: [0, 4] COM: (1222.83, 2203.29, 2.03171)
X: [2260, 2261] Y: [2136, 2137] Z: [0, 4] COM: (2261.01, 2137.37, 2.80792)'

# Compare the output with the expected output
if [ "$output_file_content" == "$expected_output_file" ] && [ "$reflections_file_content" == "$expected_reflections_file" ]; then
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