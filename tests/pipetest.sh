#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Define the filename for the named pipe relative to the script directory
PIPE="$SCRIPT_DIR/data_pipe"

# Define the path to the data directory
DATA_PATH="/dls/mx-scratch/gw56/i04-1-ins-huge/Insulin_6/Insulin_6_1.nxs"

# Cleanup function to remove the named pipe
cleanup() {
    rm -f "$PIPE"
    echo "Named pipe removed: $PIPE"
}

# Trap exit signals to perform cleanup
trap 'cleanup' EXIT

# Create the named pipe
mkfifo "$PIPE"
echo "Named pipe created: $PIPE"

# Open the named pipe for reading and writing, getting its file descriptor
exec {PIPE_FD}<>$PIPE
echo "File descriptor for the named pipe: $PIPE_FD"

# Run your C++ executable with the file descriptor of the pipe
./../build/spotfinder --images 40 --threads 40 --pipe_fd $PIPE_FD "$DATA_PATH" &

if [ $? -eq 0 ]; then
    echo "C++ program started successfully."
else
    echo "Failed to start C++ program."
    exit 1
fi

# Define the output file path
OUTPUT_FILE="$SCRIPT_DIR/output_file.json"
echo "Output file: $OUTPUT_FILE"

# Read data from the pipe and output it to a file
echo "Reading data from the pipe..."
# Read data from the pipe until EOF marker is encountered
while IFS= read -r line; do
    echo "Read line: $line"
    if [[ "$line" == "EOF" ]]; then
        echo "Received EOF marker, stopping reading."
        break
    else
        echo "$line" >> "$OUTPUT_FILE"
    fi
done <&$PIPE_FD
echo "Data read from the pipe and saved to: $OUTPUT_FILE"

# Close the file descriptor
exec {PIPE_FD}>&-
echo "File descriptor closed."

# End of script
