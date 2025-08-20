"""
A program that acts like a service to run GPU per-image spotfinding and indexing.

Usage:
python src/ffs/ssx_indexer.py /path/to/master.h5 --detector detector.json --cell 78 78 38 90 90 90 --wavelength 0.976
"""

import os
import subprocess
import sys
import time
import json
from typing import Iterator, Optional
import threading
import numpy as np
import pydantic
import ffs.index
from pathlib import Path
import gemmi
from service import DetectorGeometry
import argparse
from ssx_index import GPUIndexer
try:
    import ffbidx
except ModuleNotFoundError:
    raise RuntimeError("ffbidx not found, has the fast-feedback-indexer module been built and sourced?")

spotfinder_executable = Path.cwd() / "build/bin/spotfinder" ##FIXME assumes running from root dir - would use 'find_spotfinder' logic.

def run_spotfind_and_indexing(data_path, cell, panel, wavelength):

    indexer = GPUIndexer()
    indexer.cell = cell
    indexer.panel = panel
    indexer.wavelength = wavelength

    n_indexed = 0
    n_total = 0

    read_fd, write_fd = os.pipe()

    # Now run the spotfinder
    command = [
        str(spotfinder_executable),
        str(data_path),
        "--threads",
        str(20),
        "--pipe_fd",
        str(write_fd),
        "--output-for-index",
    ]
    print(f"Running: {' '.join(str(x) for x in command)}")

    def pipe_output(read_fd: int) -> Iterator[str]:
        """
        Generator to read from the pipe and yield the output

        Args:
            read_fd: The file descriptor for the pipe

        Yields:
            str: A line of JSON output
        """
        # Reader function
        with os.fdopen(read_fd, "r") as pipe_data:
            # Process each line of JSON output
            for line in pipe_data:
                line = line.strip()
                yield line

    def read_and_send() -> None:
        """
        Read from the pipe and send the output to the result queue

        This function is intended to be run in a separate thread

        Returns:
            None
        """
        nonlocal n_total, n_indexed
        # Read from the pipe and send to the result queue
        for line in pipe_output(read_fd):
            data = json.loads(line)
            data["file-seen-at"] = time.time()
            # XRC has one-based-indexing
            data["file-number"] += 1
            xyzobs_px = np.array(data["spot_centers"])
            indexing_result = indexer.index(xyzobs_px)
            n_total +=1
            if indexing_result.lattices:
                n_indexed += 1
                lattice = indexing_result.lattices[0]
                print(f"Indexed {lattice.n_indexed}/{int(xyzobs_px.size/3)} spots on image {data["file-number"]}")
                print(f"Image {data["file-number"]} results: {indexing_result.model_dump()}")

    start_time = time.monotonic()

    read_and_send_data_thread = threading.Thread(target=read_and_send)

    _spotfind_proc = subprocess.Popen(command, pass_fds=[write_fd])

    # Close the write end of the pipe (for this process)
    # spotfind_process will hold the write end open until it is done
    # This will allow the read end to detect the end of the output
    os.close(write_fd)
    # Start the read thread
    read_and_send_data_thread.start()
    # Wait for the process to finish
    _spotfind_proc.wait()

    # Log the duration
    duration = time.monotonic() - start_time
    print(f"Analysis complete in {duration:.1f} s")

    # Wait for the read thread to finish
    read_and_send_data_thread.join()
    print(f"Indexed {n_indexed}/{n_total} images")


def run(args):
    parser = argparse.ArgumentParser(
                        prog='ffs',
                        description='Runs standalone spotfinding and indexing of serial data using the GPU fast-feedback-indexer',
                        epilog='Text at the bottom of help')
    parser.add_argument('datafile')
    parser.add_argument('-c', '--cell', type=float, nargs=6, metavar=('a', 'b', 'c', 'alpha', 'beta', 'gamma'),
        help='Unit cell parameters: a b c alpha beta gamma')
    parser.add_argument("-w", "-λ", "--wavelength", type=float)
    parser.add_argument("--detector", help="Path to the detector model json")
    parsed = parser.parse_args(args)


    cell = gemmi.UnitCell(*parsed.cell)
    wavelength = parsed.wavelength
    with open(parsed.detector, "r") as f:
        detector = json.load(f)
    detector_geometry = DetectorGeometry(
        distance=detector["distance"],
        beam_center_x=detector["beam_center_x"],
        beam_center_y=detector["beam_center_y"],
        pixel_size_x=detector["pixel_size_x"], 
        pixel_size_y=detector["pixel_size_y"],
        image_size_x=int(detector["image_size_x"]),
        image_size_y=int(detector["image_size_y"])
    )
    if "thickness" in detector:
        detector_geometry.thickness = detector["thickness"]
    if "mu" in detector:
        detector_geometry.mu = detector["mu"]
    cell = np.reshape(np.array(cell.orth.mat, dtype="float32"), (3,3)) ## Cell as an orthogonalisation matrix
    panel = ffs.index.make_panel(
        detector_geometry.distance,
        detector_geometry.beam_center_x, detector_geometry.beam_center_y,
        detector_geometry.pixel_size_x, detector_geometry.pixel_size_y,
        detector_geometry.image_size_x, detector_geometry.image_size_y,
        detector_geometry.thickness, detector_geometry.mu)

    run_spotfind_and_indexing(parsed.datafile, cell, panel, wavelength)

if __name__ == "__main__":
    st = time.time()
    run(sys.argv[1:])
    print(f"Program time: {time.time()-st:.6f}s")