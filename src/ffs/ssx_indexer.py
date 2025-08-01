import os
import subprocess
import sys
import time
import json
from typing import Iterator, Optional
import threading
import ffbidx
import numpy as np
from dials.array_family import flex
from dxtbx import flumpy
from cctbx import uctbx
from dxtbx.serialize import load
from dxtbx.model import ExperimentList, Crystal, Experiment
from cctbx.sgtbx import space_group
from dials.algorithms.indexing import assign_indices
import pydantic

spotfinder_executable = "fast-feedback-service/build/bin/spotfinder"


class IndexedLatticeResult(pydantic.BaseModel):
    unit_cell: tuple[float, float, float, float, float, float]
    space_group: str
    n_indexed: pydantic.NonNegativeInt


class IndexingResult(pydantic.BaseModel):
    lattices: list[IndexedLatticeResult]
    n_unindexed: pydantic.NonNegativeInt

class GPUIndexer:

    def __init__(self, expts):
        self.indexer = ffbidx.Indexer(
            max_output_cells=32,
            max_spots=300,
            num_candidate_vectors=32,
            redundant_computations=True,
        )
        target_cell = uctbx.unit_cell((97,97,127,90,90,90))

        self.input_cell = np.reshape(
            np.array(target_cell.orthogonalization_matrix(), dtype="float32"), (3, 3)
        )
        self.expts = expts[0:1]
        self.n_indexed = 0
        self.n_total = 0

    def index(self, data, image_no):
        self.n_total += 1
        data = data.reshape(-1,3)
        ### >>>>> This could be replaced by a call to xyz_to_rlp
        refls = flex.reflection_table([])
        refls["xyzobs.px.value"] = flex.vec3_double(data)
        if refls.size() < 10:
            indexing_result = IndexingResult(
                lattices=[],
                n_unindexed=refls.size(),
            )
            return indexing_result
        refls["imageset_id"] = flex.int(
            refls.size(), 0
        )  # needed for centroid_px_to_mm
        refls["id"] = flex.int(
            refls.size(), 0
        )
        refls["panel"] = flex.size_t(
            refls.size(), 0
        )
        refls.centroid_px_to_mm(self.expts)
        refls.map_centroids_to_reciprocal_space(self.expts)
        rlp = np.array(flumpy.to_numpy(refls["rlp"]), dtype="float32")
        ### <<<<< This could be replaced by a call to xyz_to_rlp
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
                n_unindexed=refls.size(),
            )
            return indexing_result
        else:
            ### >>>>> This could be replaced by calls to existing ffs indexing/dx2 code
            candidate_crystal_models = []
            for index in cell_indices:
                j = 3 * index
                real_a = output_cells[:, j]
                real_b = output_cells[:, j + 1]
                real_c = output_cells[:, j + 2]
                crystal = Crystal(
                    real_a.tolist(),
                    real_b.tolist(),
                    real_c.tolist(),
                    space_group=space_group("P1"),
                )
                candidate_crystal_models.append(crystal)
            expt = self.expts[0]
            results = []
            n_indexed_per_candidate = []
            n_unindexed_per_candidate = []
            reflections = refls
            for cm in candidate_crystal_models:
                reflections["id"] = flex.int(reflections.size(), -1)
                reflections.unset_flags(flex.bool(reflections.size(), True), reflections.flags.indexed)
                experiments = ExperimentList(
                    [Experiment(
                        imageset=expt.imageset,
                        beam=expt.beam,
                        detector=expt.detector,
                        goniometer=expt.goniometer,
                        scan=expt.scan,
                        crystal=cm,
                    )]
                )
                assign_indices_ = assign_indices.AssignIndicesGlobal(tolerance=0.3)
                assign_indices_(reflections, experiments)
                n = reflections.get_flags(reflections.flags.indexed).count(True)
                n_indexed_per_candidate.append(n)
                n_unindexed_per_candidate.append(reflections.size() - n)
                results.append(experiments)
            if any(n_indexed_per_candidate):
                best = n_indexed_per_candidate.index(max(n_indexed_per_candidate))
                expt = results[best][0]
                print(f"Indexed {n_indexed_per_candidate[best]}/{reflections.size()} spots on image {image_no-1}")
                self.n_indexed += 1
                indexing_result = IndexingResult(
                    lattices=[
                        IndexedLatticeResult(
                            unit_cell=expt.crystal.get_unit_cell().parameters(),
                            space_group=str(expt.crystal.get_space_group().info()),
                            n_indexed=n_indexed_per_candidate[best],
                        )
                    ],
                    n_unindexed=n_unindexed_per_candidate[best],
                )
            else:
                indexing_result = IndexingResult(
                    lattices=[],
                    n_unindexed=reflections.size(),
                )
            ### <<<<< This could be replaced by calls to existing ffs indexing/dx2 code
            return indexing_result


    def run(self, data_path):
        read_fd, write_fd = os.pipe()

        # Now run the spotfinder
        command = [
            str(spotfinder_executable),
            str(data_path),
            "--images",
            str(100),
            "--start-index",
            str(1),
            "--threads",
            str(20),
            "--pipe_fd",
            str(write_fd),
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
                result = self.index(xyzobs_px, data["file-number"])
                #refls = flex.reflection_table([])
                #refls["xyzobs.px.mm"] = flumpy.from_numpy(xyzobs_px)
                #print(f"Sending: {data}")
                #rw.set_default_channel("result")
                #rw.send_to("result", data)

            #self.log.info("Results finished sending")

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

        # Log the duration
        duration = time.monotonic() - start_time
        print(f"Analysis complete in {duration:.1f} s")

        # Wait for the read thread to finish
        read_and_send_data_thread.join()
        print(f"Indexed {self.n_indexed}/{self.n_total} images")


def run(args):
    idxr = GPUIndexer(load.experiment_list(args[1]))
    idxr.run(args[0])

if __name__ == "__main__":
    run(sys.argv[1:])