"""
A program to run standalone indexing of strong spots from still shot images,
the equivalent of the indexing part of the ffs service.

Usage: python src/ffs/ssx_index.py -r strong.refl -e imported.expt --cell 79 79 38 90 90 90
"""

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
    U_matrix: tuple[float, float, float, float, float, float, float, float, float]
    B_matrix: tuple[float, float, float, float, float, float, float, float, float]
    space_group: str
    n_indexed: NonNegativeInt
    miller_indices: list[int] | None = None
    xyzobs_px: list[float] | None = None
    xyzcal_px: list[float] | None = None


class IndexingResult(BaseModel):
    lattices: list[IndexedLatticeResult]
    n_unindexed: NonNegativeInt


class GPUIndexer:
    def __init__(self):
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

    def index(self, xyzobs_px : np.array) -> IndexingResult:
        if xyzobs_px.size < 30:
            indexing_result = IndexingResult(
                lattices=[],
                n_unindexed=int(xyzobs_px.size / 3),
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
            min_spots=8,
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
            min_spots=8,
            method="ifssr",
        )
        if cell_indices is None:
            indexing_result = IndexingResult(
                lattices=[],
                n_unindexed=int(xyzobs_px.size / 3),
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
            ## here, we could do a stills reflection prediction to calculate rmsds.

            n_indexed, cell, orientation, B_matrix, miller_indices, s1, xyzcal_px, delpsi = (
                ffs.index.index_from_ssx_cells(cells, rlp_, xyzobs_px, np.asarray([0.0,0.0,-1.0/self.wavelength], dtype=np.float64),
                self.panel)
            )

            this_cell = gemmi.UnitCell(*cell)
            B = np.reshape(np.array(B_matrix, dtype="float64"), (3, 3))#.transpose()
            U = np.reshape(np.array(orientation, dtype="float64"), (3, 3))#.transpose()
            A = np.matmul(U, B)
            
            midx = np.array(miller_indices).reshape(-1,3)
            xyzobs = np.array(xyzobs_px).reshape(-1,3)
            xyzcal = np.array(xyzcal_px).reshape(-1,3)
            s1_reshape = np.array(s1).reshape(-1,3)
            rlp_reshape = np.array(rlp_).reshape(-1,3)
            n = 0
            rmsdx = 0
            rmsdy = 0
            rmsd_psi = 0
            print(rlp_reshape.shape[0])
            print(list(xyzcal_px))
            print(list(xyzobs_px))
            sel = np.array([False] * rlp_reshape.shape[0])
            for i, (m, obs, cal, dpsi) in enumerate(zip(midx, xyzobs, xyzcal, delpsi)):
                if m.any():
                    delta_r = ((obs[0] - cal[0])**2 + (obs[1] - cal[1])**2)**0.5
                    if delta_r < 2: # very crude outlier rejection for a starting point
                        n += 1
                        rmsdx += (obs[0] - cal[0])**2
                        rmsdy += (obs[1] - cal[1])**2
                        rmsd_psi = dpsi**2
                        sel[i] = True
            if n:
                rmsdx = (rmsdx / n)**0.5
                rmsdy = (rmsdy / n)**0.5
                rmsd_psi = (rmsd_psi / n)**0.5

            print(
                f"Indexed {(n)}/{len(rlp_) / 3} spots on image {i + 1}, RMSDs (px,px,rad): {rmsdx:.4f}, {rmsdy:.4f}, {rmsd_psi:.6f}"
            )
            print(A)
            assert 0
            
            n_unindexed = int(xyzobs_px.size / 3) - n_indexed
            indexing_result = IndexingResult(
                lattices=[
                    IndexedLatticeResult(
                        unit_cell=list(cell),
                        space_group="P1",
                        n_indexed=n_indexed,
                        U_matrix=list(orientation),
                        B_matrix=list(B_matrix),
                        miller_indices=miller_indices,
                        xyzobs_px=xyzobs_px,
                    )
                ],
                n_unindexed=n_unindexed,
            )
        return indexing_result


def run(args):
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
    parser.add_argument("--test", action="store_true", help="Run in test mode")

    parsed = parser.parse_args(args)
    if not parsed.experiments:
        print("No imported experiment list provided.")
        return
    with open(parsed.experiments, "r") as f:
        expts = json.load(f)
    wavelength = expts["beam"][0]["wavelength"]
    detector_dict = expts["detector"][0]["hierarchy"]
    panel_dict = expts["detector"][0]["panels"][
        0
    ]  # only single panel detectors for now.
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
        r = h5py.File(parsed.reflections)
        xyzs = r["dials"]["processing"]["group_0"]["xyzobs.px.value"]
        ids = r["dials"]["processing"]["group_0"]["id"]
        experiment_ids = r["dials"]["processing"]["group_0"].attrs["experiment_ids"]
        identifiers = r["dials"]["processing"]["group_0"].attrs["identifiers"]
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

    ## Split the data array into image number
    tables = []
    id_values = []
    output_crystals = {}
    for id_ in sorted(set(ids)):
        sel = ids == id_
        xyzs_this = xyzs[sel]
        if xyzs_this.any():
            tables.append(xyzs_this)
            id_values.append(id_)

    indexer = GPUIndexer()
    indexer.panel = panel
    indexer.cell = input_cell
    indexer.wavelength = wavelength

    n_indexed_images = 0
    n_total = len(tables)
    n_considered = 0

    # items to aggregate for output.
    miller_indices_output = []
    xyzobs_output = []
    xyzcal_px_output = []
    delpsical_output = []
    ids_output = []
    image_nos_output = []
    output_id = 0
    new_id_to_old_id = {}
    output_crystals_list = []
    output_crystals_id_nos = []

    for t, i in zip(tables, id_values):
        data = t.flatten()
        if t.shape[0] < 10:
            continue
        n_considered += 1
        data = t.flatten()
        result = indexer.index(data)
        if result.lattices:
            n_indexed_images += 1
            lattice = result.lattices[0]
            #print(f"Image {i + 1} results: {result.model_dump()}")
            output_crystals[identifiers_map[int(i)]] = {
                "cell": lattice.unit_cell,
                "U_matrix": lattice.U_matrix,
                "n_indexed": lattice.n_indexed,
            }
            #this_cell = gemmi.UnitCell(*lattice.unit_cell)
            B = np.reshape(np.array(lattice.B_matrix, dtype="float64"), (3, 3)).transpose()
            U = np.reshape(np.array(lattice.U_matrix, dtype="float64"), (3, 3)).transpose()
            A_inv = np.linalg.inv(np.matmul(U, B))
            output_crystals_list.append(
                {
                    "__id__": "crystal",
                    "real_space_a": [A_inv[0, 0], A_inv[0, 1], A_inv[0, 2]],
                    "real_space_b": [A_inv[1, 0], A_inv[1, 1], A_inv[1, 2]],
                    "real_space_c": [A_inv[2, 0], A_inv[2, 1], A_inv[2, 2]],
                    "space_group_hall_symbol": "P 1",
                }
            )
            output_crystals_id_nos.append(i)
            UB = np.matmul(U, B)
            #print(UB)
            s1, xyzcal_px, delpsical = ffs.index.ssx_predict(
                lattice.miller_indices,
                np.asarray([0.0,0.0,-1.0/wavelength], dtype=np.float64),
                UB,
                panel
            )
            midx = np.array(lattice.miller_indices).reshape(-1,3)
            xyzobs = np.array(lattice.xyzobs_px).reshape(-1,3)
            xyzcal = np.array(xyzcal_px).reshape(-1,3)
            s1_reshape = np.array(s1).reshape(-1,3)
            rlp_reshape = np.array(data).reshape(-1,3)
            n = 0
            rmsdx = 0
            rmsdy = 0
            rmsd_psi = 0
            sel = np.array([False] * rlp_reshape.shape[0])
            for i, (m, obs, cal, dpsi) in enumerate(zip(midx, xyzobs, xyzcal, delpsical)):
                if m.any():
                    delta_r = ((obs[0] - cal[0])**2 + (obs[1] - cal[1])**2)**0.5
                    if delta_r < 2: # very crude outlier rejection for a starting point
                        n += 1
                        rmsdx += (obs[0] - cal[0])**2
                        rmsdy += (obs[1] - cal[1])**2
                        rmsd_psi = dpsi**2
                        sel[i] = True
            if n:
                rmsdx = (rmsdx / n)**0.5
                rmsdy = (rmsdy / n)**0.5
                rmsd_psi = (rmsd_psi / n)**0.5

            print(
                f"Indexed {(n)}/{int(data.size / 3)} spots on image {i + 1}, RMSDs (px,px,rad): {rmsdx:.4f}, {rmsdy:.4f}, {rmsd_psi:.6f}"
            )
            # now save stuff for output
            miller_indices_output.append(midx[sel,:])
            xyzobs_output.append(xyzobs[sel,:])
            xyzcal_px_output.append(xyzcal[sel,:])
            delpsical_output.append(np.array(delpsical)[sel])
            ids_output.append(
                np.array(
                    [output_id] * n, dtype=np.int32
                )
            )
            image_nos_output.append(
                np.array([i] * n, dtype=np.int32)
            )
            new_id_to_old_id[output_id] = i
            output_id += 1
        else:
            print(f"No indexing solution for image {i + 1}")

    print(
        f"Indexing attempted on {n_considered}/{n_total} non-empty images with >= 10 spots"
    )
    print(f"Indexed {n_indexed_images}/{n_total} non-empty images")

    # ideally would generate an indexed.expt type file...
    # ok say we have in imported.expt, need to add crystals to indexed images
    if parsed.test:
        with open("indexed_crystals.json", "w") as f:
            json.dump(output_crystals, f, indent=2)
    else:
        expts["crystal"] = output_crystals_list
        for i, id_ in enumerate(output_crystals_id_nos):
            expts["experiment"][id_]["crystal"] = i
        with open("indexed.expt", "w") as f:
            json.dump(expts, f, indent=2)

    # output in standard dials format so that can be understood by dials.
    output_refl = h5py.File(Path.cwd() / "indexed.refl", "w")
    output_refl.create_group("dials")
    output_refl["dials"].create_group("processing")
    output_refl["dials"]["processing"].create_group("group_0")
    ids_array = np.concatenate(ids_output)
    output_refl["dials"]["processing"]["group_0"]["id"] = ids_array
    output_refl["dials"]["processing"]["group_0"]["image"] = np.concatenate(
        image_nos_output
    )
    output_refl["dials"]["processing"]["group_0"]["xyzobs.px.value"] = np.concatenate(
        xyzobs_output
    )#.reshape(-1, 3)
    output_refl["dials"]["processing"]["group_0"]["xyzcal.px"] = np.concatenate(
        xyzcal_px_output
    )#.reshape(-1, 3)
    output_refl["dials"]["processing"]["group_0"]["delpsical"] = np.concatenate(delpsical_output)
    output_refl["dials"]["processing"]["group_0"]["miller_index"] = np.concatenate(
        miller_indices_output, dtype=np.int32
    )#.reshape(-1, 3)
    sorted_ids = sorted(list(set(np.uint(i) for i in new_id_to_old_id.keys())))
    output_refl["dials"]["processing"]["group_0"].attrs["experiment_ids"] = sorted_ids
    identifiers = [identifiers_map[new_id_to_old_id[i]] for i in sorted_ids]
    output_refl["dials"]["processing"]["group_0"].attrs["identifiers"] = identifiers

    output_refl["dials"]["processing"]["group_0"]["panel"] = np.zeros_like(ids_array, dtype=np.uint)
    # output_refl["dials"]["processing"]["group_0"]["xyzcal.px"] = output_refl["dials"]["processing"]["group_0"]["xyzobs.px.value"]
    ## extra potential data to output to enable further processing:
    ## rlp, panel, flags, s1, xyzcal, xyzobs.mm.value


if __name__ == "__main__":
    st = time.time()
    run(sys.argv[1:])
    print(f"Program time: {time.time() - st:.6f}s")
