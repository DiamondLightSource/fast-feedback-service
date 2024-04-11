import os
import subprocess


def run_executable(executable_path, data_filepath, num_threads, num_images):
    # Create a pipe to capture the output
    read_fd, write_fd = os.pipe()

    print("Pipe FDs:", read_fd, write_fd)

    # Command to run the executable
    command = [
        "--sample",
        data_filepath,
        "--images",
        str(num_images),
        "--threads",
        str(num_threads),
        "--pipe_fd",
        str(write_fd),
    ]

    # Run the executable
    print("Running:", " ".join(command))
    process = subprocess.Popen(command, executable=executable_path, pass_fds=[write_fd])

    # Read from the pipe
    with os.fdopen(read_fd, "r") as pipe_in_file:
        for line in pipe_in_file:
            # Process each line of JSON output
            if line.strip() == "EOF":
                print("End of output")
                break  # Exit the loop when "EOF" is received
            print("Received:", line.strip())

    # Wait for the process to finish
    process.wait()


if __name__ == "__main__":
    executable_path = "../build/spotfinder"
    data_filepath = "/dls/mx-scratch/gw56/i04-1-ins-huge/Insulin_6/Insulin_6_1.nxs"
    num_threads = 40  # Set the number of threads as needed
    num_images = 40  # Set the number of images as needed

    run_executable(executable_path, data_filepath, num_threads, num_images)
