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

try:
    import ffbidx
except ModuleNotFoundError:
    raise RuntimeError("ffbidx not found, has the fast-feedback-indexer module been built and sourced?")

spotfinder_executable = Path.cwd() / "build_ws448/bin/spotfinder" ##FIXME assumes running from root dir - would use 'find_spotfinder' logic.

class IndexedLatticeResult(pydantic.BaseModel):
    unit_cell: tuple[float, float, float, float, float, float]
    U_matrix: tuple[float, float, float, float, float, float, float, float, float]
    space_group: str
    n_indexed: pydantic.NonNegativeInt

class IndexingResult(pydantic.BaseModel):
    lattices: list[IndexedLatticeResult]
    n_unindexed: pydantic.NonNegativeInt

class GPUIndexer:

    def __init__(self, cell: gemmi.UnitCell, detector: DetectorGeometry, wavelength:float):
        self.indexer = ffbidx.Indexer(
            max_output_cells=32,
            max_spots=300,
            num_candidate_vectors=32,
            redundant_computations=True,
        )
        self.panel = ffs.index.make_panel(
            detector.distance,
            detector.beam_center_x, detector.beam_center_y,
            detector.pixel_size_x, detector.pixel_size_y,
            detector.image_size_x, detector.image_size_y
        )
        self.input_cell = np.reshape(np.array(cell.orth.mat, dtype="float32"), (3,3)) ## As an orthogonalisation matrix
        self.n_indexed = 0
        self.n_total = 0
        self.wavelength = wavelength

    def index(self, data, image_no):
        self.n_total += 1
        if data.size < 30:
            indexing_result = IndexingResult(
                lattices=[],
                n_unindexed=int(data.size/3),
            )
            return indexing_result

        rlp_ = ffs.index.ssx_xyz_to_rlp(data, self.wavelength, self.panel) ## FIXME pass through wavelength.
        rlp = np.array(rlp_, dtype="float32").reshape(-1,3)
        rlp = rlp.transpose().copy()

        output_cells, scores = self.indexer.run(
            rlp,
            self.input_cell,
            dist1=0.3,
            dist3=0.15,
            num_halfsphere_points=32768,
            max_dist=0.00075,
            min_spots=8,
            n_output_cells=32,
            method="ifssr",
            triml=0.001,
            trimh=0.3,
            delta=0.1,
        )

        cell_indices = self.indexer.crystals(
            output_cells,
            rlp,
            scores,
            threshold=0.00075,
            min_spots=8,
            method="ifssr",
        )
        if cell_indices is None:
            indexing_result = IndexingResult(
                lattices=[],
                n_unindexed=int(data.size/3),
            )
        else:
            cells = np.array([], dtype="float64")
            for index in cell_indices:
                j = 3 * index
                real_a = output_cells[:, j]
                real_b = output_cells[:, j + 1]
                real_c = output_cells[:, j + 2]
                cells = np.concatenate((cells, real_a, real_b, real_c), axis=None)
            ## For now, just determines the cell with the highest number of indexed spots.
            n_indexed, cell, orientation = ffs.index.index_from_ssx_cells(cells, rlp_, data)
            n_unindexed = int(data.size/3) - n_indexed
            print(f"Indexed {n_indexed}/{int(data.size/3)} spots on image {image_no}")
            indexing_result = IndexingResult(
                lattices=[
                    IndexedLatticeResult(
                        unit_cell=list(cell),
                        space_group="P1",
                        n_indexed=n_indexed,
                        U_matrix=list(orientation),
                    )
                ],
                n_unindexed=n_unindexed,
            )
            print(f"Image {image_no} results: {indexing_result.model_dump()}")
            self.n_indexed += 1
        return indexing_result


    def run(self, data_path):
        read_fd, write_fd = os.pipe()

        # Now run the spotfinder
        command = [
            str(spotfinder_executable),
            str(data_path),
            "--threads",
            str(20),
            "--pipe_fd",
            str(write_fd),
            "--pipe-output-for-index",
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
            # Read from the pipe and send to the result queue
            for line in pipe_output(read_fd):
                data = json.loads(line)
                data["file-seen-at"] = time.time()
                # XRC has one-based-indexing
                data["file-number"] += 1
                xyzobs_px = np.array(data["spot_centers"])
                indexing_result = self.index(xyzobs_px, data["file-number"])
                '''# Pass through all file* fields
                for key in (x for x in message if x.startswith("file")):
                    result[key] = message[key]
                self.log.info(f"{message=}")
                self.log.info(f"{result=}")
                # Send results onwards
                rw.set_default_channel("result")
                rw.send_to("result", result, transaction=txn)
                rw.transport.transaction_commit(txn)'''

        start_time = time.monotonic()

        read_and_send_data_thread = threading.Thread(target=read_and_send)

        self._spotfind_proc = subprocess.Popen(command, pass_fds=[write_fd])

        # Close the write end of the pipe (for this process)
        # spotfind_process will hold the write end open until it is done
        # This will allow the read end to detect the end of the output
        os.close(write_fd)
        # Start the read thread
        read_and_send_data_thread.start()
        # Wait for the process to finish
        self._spotfind_proc.wait()

        #FIXME seems to have an issue of missing the last few if the spotfinder ends prematurely
        # sometimes see Error: Could not find data file for frame 0?
        time.sleep(0.05)
        # Log the duration
        duration = time.monotonic() - start_time
        print(f"Analysis complete in {duration:.1f} s")

        # Wait for the read thread to finish
        read_and_send_data_thread.join()
        print(f"Indexed {self.n_indexed}/{self.n_total} images")


def run(args):
    print(args)
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

    idxr = GPUIndexer(cell, detector_geometry, wavelength)
    idxr.run(parsed.datafile)

if __name__ == "__main__":
    st = time.time()
    run(sys.argv[1:])
    print(f"Program time: {time.time()-st:.6f}s")