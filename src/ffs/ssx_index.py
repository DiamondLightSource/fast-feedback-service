"""
A program to run standalone indexing of strong spots from still shot images,
the equivalent of the indexing part of the ffs service.

Usage: python src/ffs/ssx_index.py -r strong.refl --detector detector.json --cell 78 78 38 90 90 90 --wavelength 0.976
"""
import argparse
import sys
import h5py
import gemmi
import numpy as np
import json
import ffs.index
from pydantic import BaseModel, NonNegativeInt
import time


class IndexedLatticeResult(BaseModel):
    unit_cell: tuple[float, float, float, float, float, float]
    U_matrix: tuple[float, float, float, float, float, float, float, float, float]
    space_group: str
    n_indexed: NonNegativeInt


class IndexingResult(BaseModel):
    lattices: list[IndexedLatticeResult]
    n_unindexed: NonNegativeInt

class GPUIndexer():

    def __init__(self):
        import ffbidx
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

    def index(self, xyzobs_px):
        if xyzobs_px.size < 30:
            indexing_result = IndexingResult(
                lattices=[],
                n_unindexed=int(xyzobs_px.size/3),
            )
            return indexing_result
        rlp_ = ffs.index.ssx_xyz_to_rlp(xyzobs_px, self.wavelength, self.panel)
        rlp = np.array(rlp_, dtype="float32").reshape(-1,3)
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
                n_unindexed=int(xyzobs_px.size/3),
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
            n_indexed, cell, orientation = ffs.index.index_from_ssx_cells(cells, rlp_, xyzobs_px)
            n_unindexed = int(xyzobs_px.size/3) - n_indexed
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
        return indexing_result

def run(args):
    parser = argparse.ArgumentParser(
                        prog='index',
                        description='Runs standalone indexing of serial data using the GPU fast-feedback-indexer',
                        epilog='Text at the bottom of help')
    parser.add_argument('-r', '--reflections')
    parser.add_argument('-c', '--cell', type=float, nargs=6, metavar=('a', 'b', 'c', 'alpha', 'beta', 'gamma'),
        help='Unit cell parameters: a b c alpha beta gamma')
    parser.add_argument("-w", "-λ", "--wavelength", type=float)
    parser.add_argument("--detector", help="Path to the detector model json")
    parsed = parser.parse_args(args)
    
    cell = gemmi.UnitCell(*parsed.cell)
    input_cell = np.reshape(np.array(cell.orth.mat, dtype="float32"), (3,3)) ## As an orthogonalisation matrix.
    wavelength = parsed.wavelength
    with open(parsed.detector, "r") as f:
        detector = json.load(f)

    try:
        r = h5py.File(parsed.reflections)
        xyzs = r["dials"]["processing"]["group_0"]["xyzobs.px.value"]
        ids = r["dials"]["processing"]["group_0"]["id"]
        experiment_ids = r["dials"]["processing"]["group_0"].attrs['experiment_ids']
        identifiers = r["dials"]["processing"]["group_0"].attrs['identifiers']
        identifiers_map = dict(zip(experiment_ids, identifiers))
    except Exception as e:
        print(f"Unable to interpret the reflection file - please check input.\n Error: {e}")
        return

    try:
        panel = ffs.index.make_panel(
            detector["distance"],
            detector["beam_center_x"], detector["beam_center_y"],
            detector["pixel_size_x"], detector["pixel_size_y"],detector["image_size_x"],
            detector["image_size_y"], detector["thickness"], detector["mu"]
        )
    except Exception as e:
        print(f"Unable to compose a detector panel model from the detector json.\n Error: {e}")
        return

    ## Split the data array into image number
    tables = []
    id_values = []
    output_crystals = {}
    for id_ in sorted(set(ids)):
        sel = (ids == id_)
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
    for t, i in zip(tables, id_values):
        data = t.flatten()
        if t.shape[0] < 10:
            continue
        data = t.flatten()
        result = indexer.index(data)
        if result.lattices:
            n_indexed_images += 1
            lattice = result.lattices[0]
            print(f"Indexed {lattice.n_indexed}/{int(data.size/3)} spots on image {i+1}")
            print(f"Image {i+1} results: {result.model_dump()}")
            output_crystals[identifiers_map[i]] = {
                "cell" : lattice.unit_cell,
                "U_matrix" : lattice.U_matrix,
                "n_indexed":lattice.n_indexed,
            }

    print(f"Indexed {n_indexed_images}/{n_total} images")
    with open("indexed_crystals.json", "w") as f:
        json.dump(output_crystals, f, indent=2)

if __name__ == "__main__":
    st = time.time()
    run(sys.argv[1:])
    print(f"Program time: {time.time()-st:.6f}s")