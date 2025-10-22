"""
A program to run standalone indexing of strong spots from still shot images.

Usage: ./build/bin/ssx_index -r results_ffs.h5 -e imported.expt -c 79 79 38 90 90 90
"""
# ruff: noqa: C901

import argparse
import json
import sys
import time
from pathlib import Path

import gemmi
import h5py
import numpy as np
from pydantic import BaseModel, NonNegativeInt

import ffs.index


class IndexedLatticeResult(BaseModel):
    unit_cell: tuple[float, float, float, float, float, float]
    A_matrix: tuple[float, float, float, float, float, float, float, float, float]
    space_group: str
    n_indexed: NonNegativeInt
    miller_indices: list[int] | None = None
    xyzobs_px: list[float] | None = None
    xyzcal_px: list[float] | None = None
    s1: list[float] | None = None
    delpsi: list[float] | None = None
    rmsds: list[float] | None = None


class IndexingResult(BaseModel):
    lattices: list[IndexedLatticeResult]
    n_unindexed: NonNegativeInt


class GPUIndexer:
    def __init__(self, min_spots=10):
        import ffbidx  # Delay import to here, so that the fast
        # feedback service does not need this available to run.

        self._indexer = ffbidx.Indexer(
            max_output_cells=32,
            max_spots=300,
            num_candidate_vectors=32,
            redundant_computations=True,
        )
        ## We won't be able to set these until we get a PIA request.
        self._panel = None
        self._cell = None
        self._wavelength = None
        self._s0 = None
        self.min_spots = min_spots

    @property
    def cell(self):
        return self._cell

    @cell.setter
    def cell(self, new_cell):
        self._cell = new_cell

    @property
    def panel(self):
        return self._panel

    @panel.setter
    def panel(self, new_panel):
        self._panel = new_panel

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, new_wavelength):
        self._wavelength = new_wavelength
        self._s0 = np.asarray([0.0, 0.0, -1.0 / self._wavelength], dtype=np.float64)

    @property
    def s0(self):
        return self._s0

    def index(self, xyzobs_px: np.array) -> IndexingResult:
        n_initial = int(xyzobs_px.size / 3)
        if xyzobs_px.size < (self.min_spots * 3):
            indexing_result = IndexingResult(
                lattices=[],
                n_unindexed=n_initial,
            )
            return indexing_result

        rlp_ = ffs.index.ssx_xyz_to_rlp(xyzobs_px, self.wavelength, self.panel)
        rlp = np.array(rlp_, dtype="float32").reshape(-1, 3)
        rlp = rlp.transpose().copy()

        output_cells, scores = self._indexer.run(
            rlp,
            self.cell,
            dist1=0.3,
            dist3=0.15,
            num_halfsphere_points=32768,
            max_dist=0.00075,
            min_spots=max(6, self.min_spots - 2),  # dials default
            n_output_cells=32,
            method="ifssr",
            triml=0.001,
            trimh=0.3,
            delta=0.1,
        )
        cell_indices = self._indexer.crystals(
            output_cells,
            rlp,
            scores,
            threshold=0.00075,
            min_spots=max(6, self.min_spots - 2),  # dials default
            method="ifssr",
        )
        if cell_indices is None:
            indexing_result = IndexingResult(
                lattices=[],
                n_unindexed=n_initial,
            )
        else:
            cells = np.array([], dtype="float64")
            for index in cell_indices:
                j = 3 * index
                real_a = output_cells[:, j]
                real_b = output_cells[:, j + 1]
                real_c = output_cells[:, j + 2]
                cells = np.concatenate((cells, real_a, real_b, real_c), axis=None)

            ssx_index_result = (
                ffs.index.index_from_ssx_cells(cells, rlp_, xyzobs_px, self.s0, self.panel)
            )
            n_indexed = len(ssx_index_result.delpsi)

            n_unindexed = n_initial - n_indexed
            indexing_result = IndexingResult(
                lattices=[
                    IndexedLatticeResult(
                        unit_cell=list(ssx_index_result.cell_parameters),
                        space_group="P1",
                        n_indexed=n_indexed,
                        A_matrix=ssx_index_result.A_matrix.flatten(),
                        miller_indices=ssx_index_result.miller_indices,
                        xyzobs_px=ssx_index_result.xyzobs_px,
                        xyzcal_px=ssx_index_result.xyzcal_px,
                        s1=ssx_index_result.s1,
                        delpsi=ssx_index_result.delpsi,
                        rmsds=ssx_index_result.rmsds
                    )
                ],
                n_unindexed=n_unindexed,
            )
        return indexing_result


class OutputAggregator:
    """
    Helper class to aggregate per-image indexing to output as as reflection table and
    an experiment list.
    """

    def __init__(self, identifiers_map):
        # Lists of numpy data arrays to be concatenated.
        self.miller_indices_output = []
        self.xyzobs_output = []
        self.xyzcal_px_output = []
        self.delpsical_output = []
        self.ids_output = []
        self.s1_output = []
        self.image_nos_output = []
        # crystal data to output.
        self.output_id = 0
        self.new_id_to_old_id = {}
        self.output_crystals_list = []
        self.output_crystals_id_nos = []
        self.identifiers_map = identifiers_map

    def add_result(self, lattice, i):
        A = np.reshape(np.array(lattice.A_matrix, dtype="float64"), (3, 3))
        A_inv = np.linalg.inv(A)
        self.output_crystals_list.append(
            {
                "__id__": "crystal",
                "real_space_a": [A_inv[0, 0], A_inv[0, 1], A_inv[0, 2]],
                "real_space_b": [A_inv[1, 0], A_inv[1, 1], A_inv[1, 2]],
                "real_space_c": [A_inv[2, 0], A_inv[2, 1], A_inv[2, 2]],
                "space_group_hall_symbol": "P 1",
            }
        )
        self.output_crystals_id_nos.append(i)
        midx = np.array(lattice.miller_indices, dtype=int).reshape(-1, 3)
        xyzobs = np.array(lattice.xyzobs_px).reshape(-1, 3)
        xyzcal = np.array(lattice.xyzcal_px).reshape(-1, 3)
        s1_reshape = np.array(lattice.s1).reshape(-1, 3)
        delpsi = np.array(lattice.delpsi)
        rmsdx, rmsdy, rmsd_psi = lattice.rmsds

        n = xyzcal.shape[0]

        # now save for output
        self.miller_indices_output.append(midx)
        self.xyzobs_output.append(xyzobs)
        self.xyzcal_px_output.append(xyzcal)
        self.delpsical_output.append(np.array(delpsi))
        self.s1_output.append(s1_reshape)
        self.ids_output.append(np.full(n, self.output_id, dtype=np.int32))
        self.image_nos_output.append(np.full(n, i, dtype=np.int32))
        self.new_id_to_old_id[self.output_id] = i
        self.output_id += 1

    def write_table(self, filename):
        with h5py.File(Path.cwd() / filename, "w") as output_refl:
            group = output_refl.create_group("dials/processing/group_0")
            ids_array = np.concatenate(self.ids_output)
            group["id"] = ids_array
            group["image"] = np.concatenate(self.image_nos_output)
            group["xyzobs.px.value"] = np.concatenate(self.xyzobs_output)
            group["xyzcal.px"] = np.concatenate(self.xyzcal_px_output)
            group["s1"] = np.concatenate(self.s1_output)
            group["delpsical.rad"] = np.concatenate(self.delpsical_output)
            group["miller_index"] = np.concatenate(self.miller_indices_output, dtype=np.int32)
            sorted_ids = sorted(list(set(np.uint(i) for i in self.new_id_to_old_id.keys())))
            group.attrs["experiment_ids"] = sorted_ids
            identifiers = [self.identifiers_map[self.new_id_to_old_id[i]] for i in sorted_ids]
            group.attrs["identifiers"] = identifiers
            group["panel"] = np.zeros_like(ids_array, dtype=np.uint)
            ## extra potential data to output to enable further processing:
            ## rlp, flags, xyzobs.mm.value


def run(args):
    st = time.time()
    parser = argparse.ArgumentParser(
        prog="index",
        description="Runs standalone indexing of serial data using the GPU fast-feedback-indexer",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("-r", "--reflections", help="Path to the strong spots h5 file")
    parser.add_argument("-e", "--experiments", help="Path to the imported.expt json")
    parser.add_argument(
        "-c",
        "--cell",
        type=float,
        nargs=6,
        metavar=("a", "b", "c", "alpha", "beta", "gamma"),
        help="Unit cell parameters: a b c alpha beta gamma",
    )
    parser.add_argument(
        "--min-spots", type=int, default=10, help="Only attempt indexing on"
    )
    parser.add_argument("--test", action="store_true", help="Run in test mode")

    parsed = parser.parse_args(args)
    min_spots = parsed.min_spots
    if not parsed.experiments:
        print("No imported experiment list provided.")
        return
    with open(parsed.experiments, "r") as f:
        expts = json.load(f)
    wavelength = expts["beam"][0]["wavelength"]
    detector_dict = expts["detector"][0]["hierarchy"]
    # only single panel detectors for now.
    panel_dict = expts["detector"][0]["panels"][0]
    detector = {
        "distance": -1.0 * (detector_dict["origin"][2] + panel_dict["origin"][2]),
        "beam_center_x": -1.0
        * (detector_dict["origin"][0] + panel_dict["origin"][0])
        / panel_dict["pixel_size"][0],
        "beam_center_y": (detector_dict["origin"][1] + panel_dict["origin"][1])
        / panel_dict["pixel_size"][1],
        "pixel_size_x": panel_dict["pixel_size"][0],
        "pixel_size_y": panel_dict["pixel_size"][1],
        "image_size_x": panel_dict["image_size"][0],
        "image_size_y": panel_dict["image_size"][1],
        "thickness": panel_dict["thickness"],
        "mu": panel_dict["mu"],
    }

    cell = gemmi.UnitCell(*parsed.cell)
    input_cell = np.reshape(
        np.array(cell.orth.mat, dtype="float32"), (3, 3)
    )  ## As an orthogonalisation matrix.

    if not parsed.reflections:
        print("No strong reflections h5 file provided.")
        return
    try:
        with h5py.File(parsed.reflections, "r") as refls:
            processing_group = refls["dials"]["processing"]["group_0"]
            xyzs = processing_group["xyzobs.px.value"][:]
            ids = processing_group["id"][:]
            experiment_ids = processing_group.attrs["experiment_ids"]
            identifiers = processing_group.attrs["identifiers"]
            identifiers_map = dict(zip(experiment_ids, identifiers))
    except Exception as e:
        print(
            f"Unable to interpret the reflection file - please check input.\n Error: {e}"
        )
        return

    try:
        panel = ffs.index.make_panel(
            detector["distance"],
            detector["beam_center_x"],
            detector["beam_center_y"],
            detector["pixel_size_x"],
            detector["pixel_size_y"],
            detector["image_size_x"],
            detector["image_size_y"],
            detector["thickness"],
            detector["mu"],
        )
    except Exception as e:
        print(
            f"Unable to compose a detector panel model from the detector json.\n Error: {e}"
        )
        return

    output_aggregator = OutputAggregator(identifiers_map)
    tables = []
    id_values = []

    ## Note this assumes ids are in ascending order, which is the
    ## expected form of the output from spotfinding.
    unique_ids, start_indices = np.unique(ids, return_index=True)
    end_indices = np.append(start_indices[1:], len(ids))
    for id_, start, end in zip(unique_ids, start_indices, end_indices):
        xyzs_this = xyzs[start:end]
        if xyzs_this.any():
            tables.append(xyzs_this)
            id_values.append(id_)

    ## Initialise the GPU indexer.
    try:
        indexer = GPUIndexer()
    except ModuleNotFoundError as e:  # if ffbidx not sourced
        print(f"ModuleNotFoundError: {e}")
        print(
            "ffbidx not found, has the fast-feedback-indexer module been built and sourced?"
        )
        return
    except ImportError as e:
        print(f"ImportError: {e}")
        print(
            "Potential source of this error: has the CUDA Runtime Library been loaded?"
        )
        return
    indexer.panel = panel
    indexer.cell = input_cell
    indexer.wavelength = wavelength

    # Quantities to log for log output
    n_indexed_images = 0
    n_total = len(tables)
    n_considered = 0

    t1 = time.time()

    for t, i in zip(tables, id_values):
        data = t.flatten()
        if t.shape[0] < min_spots:
            continue
        n_considered += 1
        data = t.flatten()
        result = indexer.index(data)
        if result.lattices:
            n_indexed_images += 1
            lattice = result.lattices[0]
            # now save stuff for output
            output_aggregator.add_result(lattice, i)
            # print number of spots indexed and rmsds
            rmsdx, rmsdy, rmsd_psi = lattice.rmsds
            cell_str = ", ".join(f"{i:.3f}" for i in lattice.unit_cell)
            print(
                f"Indexed {(lattice.n_indexed)}/{int(data.size / 3)} spots on image {i + 1}:\n"
                + f"  cell: {cell_str}\n"
                + f"  RMSDs: (x(px), y(px), psi(rad)): {rmsdx:.3f}, {rmsdy:.3f}, {rmsd_psi:.5f}"
            )
        else:
            print(f"No indexing solution for image {i + 1}")

    t2 = time.time()
    print(
        f"Indexing attempted on {n_considered}/{n_total} non-empty images with >= {min_spots} spots"
    )
    print(f"Indexed {n_indexed_images}/{n_total} non-empty images in {t2 - t1:.3f}s")

    # ideally would generate an indexed.expt type file...
    # ok say we have in imported.expt, need to add crystals to indexed images
    if parsed.test:
        with open("indexed_crystals.json", "w") as f:
            json.dump(output_aggregator.output_crystals_list, f, indent=2)
    else:
        expts["crystal"] = output_aggregator.output_crystals_list
        for i, id_ in enumerate(output_aggregator.output_crystals_id_nos):
            expts["experiment"][id_]["crystal"] = i
        with open("indexed.expt", "w") as f:
            json.dump(expts, f, indent=2)

    # output in standard dials format so that can be understood by dials.
    if not output_aggregator.ids_output:
        print("No images successfully indexed, no reflection output will be written.")
    else:
        output_aggregator.write_table("indexed.refl")
    t3 = time.time()
    print(
        f"Setup time: {t1 - st:.3f}s, index time {t2 - t1:.3f}s, write time {t3 - t2:.3f}s"
    )


if __name__ == "__main__":
    st = time.time()
    run(sys.argv[1:])
    print(f"Program time: {time.time() - st:.3f}s")
