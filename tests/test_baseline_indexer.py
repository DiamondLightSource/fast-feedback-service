import json
import os
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np


def test_baseline_indexer(tmp_path, dials_data):
    indexer_path: str | Path | None = os.getenv("INDEXER")
    assert indexer_path is not None
    source = dials_data("indexing_test_data", pathlib=True) / "ins14_24_strong.refl.gz"
    expts = dials_data("indexing_test_data", pathlib=True) / "ins14_24_imported.expt"
    shutil.copy(source, tmp_path / source.name)
    subprocess.run(["gunzip", source.name], cwd=tmp_path)
    subprocess.run(
        [
            indexer_path,
            "-r",
            "ins14_24_strong.refl",
            "-e",
            os.fspath(expts),
            "--max-cell",
            "100",
            "--dmin",
            "1.81",
            "--max-refine",
            "5",
            "--test",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    with open(tmp_path / "candidate_vectors.json", "r") as f:
        candidate_vectors = json.load(f)
    expected_output = {
        "00": [-0.38288460328028456, -16.046345646564774, 2.9586537526204055],
        "01": [-41.00043348644091, 6.374347624571426, 53.04086788840915],
        "02": [-25.233528614044186, -61.114115715026855, 14.959117174148561],
        "03": [-15.894061997532845, 63.236873000860214, 38.00999879837036],
        "04": [-0.45249998569488525, -13.574999570846558, 8.144999742507935],
        "05": [-18.778749406337738, -87.10624724626541, -90.95249712467194],
        "06": [-75.11499762535095, -18.09999942779541, 0.0],
        "07": [-8.144999742507935, -90.04749715328217, -25.339999198913574],
        "08": [-59.72999811172485, -82.35499739646912, -37.33124881982803],
        "09": [0.0, 1.809999942779541, -9.954999685287476],
        "10": [1.809999942779541, 26.244999170303345, 36.19999885559082],
    }
    assert candidate_vectors == expected_output

    with open(tmp_path / "candidate_crystals.json", "r") as f:
        candidate_crystals = json.load(f)
    expected_crystals_output = {
        "0": {
            "crystal": {
                "__id__": "crystal",
                "real_space_a": [
                    -0.19426335071900225,
                    2.7801174067467365,
                    6.051001054208426,
                ],
                "real_space_b": [
                    0.08619117365755113,
                    -15.44256627332296,
                    8.780280738821919,
                ],
                "real_space_c": [
                    26.776181405061205,
                    -0.4458485817621857,
                    2.390114898090674,
                ],
                "space_group_hall_symbol": "P 1",
            },
            "fraction_indexed": 0.10682492581602374,
            "indexed_score": 0.6629650127224291,
            "num_indexed": 36.0,
            "rmsd_score": 2.2201939081181536,
            "rmsdxy": 3.380438094882832,
            "score": 2.883158920840583,
            "volume_score": 0.0,
        },
        "1": {
            "crystal": {
                "__id__": "crystal",
                "real_space_a": [
                    0.4481316974199212,
                    15.415927201656164,
                    -3.0275331516335937,
                ],
                "real_space_b": [
                    -23.669999591400334,
                    2.451655058915848,
                    3.0805873818471756,
                ],
                "real_space_c": [
                    6.859148012753957,
                    10.20967906616398,
                    43.357950357525134,
                ],
                "space_group_hall_symbol": "P 1",
            },
            "fraction_indexed": 0.13353115727002968,
            "indexed_score": 0.34103691783506696,
            "num_indexed": 45.0,
            "rmsd_score": 0.862557286804198,
            "rmsdxy": 1.3191179831556559,
            "score": 3.6274071965532717,
            "volume_score": 2.423812991914007,
        },
        "2": {
            "crystal": {
                "__id__": "crystal",
                "real_space_a": [
                    -0.4068140301237836,
                    -14.2335896432367,
                    8.564555114263998,
                ],
                "real_space_b": [
                    -22.948009311618886,
                    -7.172171064419543,
                    -16.887995768461504,
                ],
                "real_space_c": [
                    40.19545822862181,
                    -20.374384894187973,
                    -43.00228376073281,
                ],
                "space_group_hall_symbol": "P 1",
            },
            "fraction_indexed": 0.08902077151335312,
            "indexed_score": 0.9259994185562226,
            "num_indexed": 30.0,
            "rmsd_score": 0.5276192295238264,
            "rmsdxy": 1.0458207131523087,
            "score": 4.709400506327247,
            "volume_score": 3.2557818582471985,
        },
        "3": {
            "crystal": {
                "__id__": "crystal",
                "real_space_a": [
                    0.37266400808892014,
                    16.230354909114038,
                    -3.033100126735198,
                ],
                "real_space_b": [
                    40.74989194614894,
                    -6.681288152933564,
                    -52.74916747787913,
                ],
                "real_space_c": [
                    -57.517040647405615,
                    -1.0771626705005324,
                    -52.54106573437347,
                ],
                "space_group_hall_symbol": "P 1",
            },
            "fraction_indexed": 0.16913946587537093,
            "indexed_score": 0.0,
            "num_indexed": 57.0,
            "rmsd_score": 0.0,
            "rmsdxy": 0.7254843104456131,
            "score": 4.755675887505278,
            "volume_score": 4.755675887505278,
        },
        "4": {
            "crystal": {
                "__id__": "crystal",
                "real_space_a": [
                    -0.414850669348278,
                    -13.79394632605264,
                    8.213075956315434,
                ],
                "real_space_b": [
                    -23.24216930086044,
                    -2.795651431456891,
                    -17.425860500421777,
                ],
                "real_space_c": [
                    40.40890491894017,
                    -19.62342856433785,
                    -44.5481398169253,
                ],
                "space_group_hall_symbol": "P 1",
            },
            "fraction_indexed": 0.0830860534124629,
            "indexed_score": 1.0255350921071376,
            "num_indexed": 28.0,
            "rmsd_score": 0.7329675955857827,
            "rmsdxy": 1.205794384007887,
            "score": 4.943956407128037,
            "volume_score": 3.1854537194351167,
        },
    }
    assert candidate_crystals == expected_crystals_output

    assert (tmp_path / "indexed.refl").is_file()
    assert (tmp_path / "indexed.expt").is_file()
    with h5py.File(tmp_path / "indexed.refl") as f:
        flags = f["/dials/processing/group_0/flags"]
        assert len(flags) == 703
        n_indexed = np.sum(np.array(flags, dtype=int) == 36)
        n_unindexed = np.sum(np.array(flags, dtype=int) == 32)
        assert n_indexed == 55
        assert n_unindexed == 648


def test_baseline_indexer_c2sum(tmp_path, dials_data):
    indexer_path: str | Path | None = os.getenv("INDEXER")
    assert indexer_path is not None
    source = dials_data("indexing_test_data", pathlib=True) / "c2sum_strong.refl.gz"
    expts = dials_data("indexing_test_data", pathlib=True) / "c2sum_imported.expt"
    shutil.copy(source, tmp_path / source.name)
    subprocess.run(["gunzip", source.name], cwd=tmp_path)
    subprocess.run(
        [
            indexer_path,
            "-r",
            "c2sum_strong.refl",
            "-e",
            os.fspath(expts),
            "--max-cell",
            "94.4",
            "--dmin",
            "1.84",
            "--max-refine",
            "5",
            "--test",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    with open(tmp_path / "candidate_crystals.json", "r") as f:
        candidate_crystals = json.load(f)

    expected_crystals_output = {
        "0": {
            "crystal": {
                "__id__": "crystal",
                "real_space_a": [
                    34.17640573400373,
                    -7.465908236923297,
                    20.499161034425466,
                ],
                "real_space_b": [
                    16.648886142399864,
                    8.075218184328115,
                    -36.086479295366246,
                ],
                "real_space_c": [
                    0.90929797277945,
                    66.87063326644437,
                    18.121049774287396,
                ],
                "space_group_hall_symbol": "P 1",
            },
            "fraction_indexed": 0.9792395858365287,
            "indexed_score": 0.005009416169085765,
            "num_indexed": 93724.0,
            "rmsd_score": 0.0,
            "rmsdxy": 0.07315121148177929,
            "score": 0.005182182229071939,
            "volume_score": 0.00017276605998617356,
        },
        "1": {
            "crystal": {
                "__id__": "crystal",
                "real_space_a": [
                    34.183737336394486,
                    -7.466708160824077,
                    20.504737658391274,
                ],
                "real_space_b": [
                    16.651981990166604,
                    8.074837535667207,
                    -36.094844734749685,
                ],
                "real_space_c": [
                    0.9088268074740147,
                    66.88708059063406,
                    18.120768250267105,
                ],
                "space_group_hall_symbol": "P 1",
            },
            "fraction_indexed": 0.9822172999968656,
            "indexed_score": 0.0006290631941424409,
            "num_indexed": 94009.0,
            "rmsd_score": 0.010656318685108968,
            "rmsdxy": 0.07369353583507099,
            "score": 0.01241280945645382,
            "volume_score": 0.0011274275772024112,
        },
        "2": {
            "crystal": {
                "__id__": "crystal",
                "real_space_a": [
                    34.174144922093575,
                    -7.465314578907954,
                    20.498111897894855,
                ],
                "real_space_b": [
                    16.649639413574143,
                    8.073981992730776,
                    -36.08565154324145,
                ],
                "real_space_c": [
                    0.9062186699312748,
                    66.86731447368523,
                    18.122817840626766,
                ],
                "space_group_hall_symbol": "P 1",
            },
            "fraction_indexed": 0.9758334987618978,
            "indexed_score": 0.010036286971724262,
            "num_indexed": 93398.0,
            "rmsd_score": 0.012429125731538093,
            "rmsdxy": 0.07378414730719672,
            "score": 0.022465412703262355,
            "volume_score": 0.0,
        },
        "3": {
            "crystal": {
                "__id__": "crystal",
                "real_space_a": [
                    -16.649288830861,
                    -8.071480003169183,
                    36.09015283684615,
                ],
                "real_space_b": [
                    -34.18080008840855,
                    7.4627986369941794,
                    -20.50092030704733,
                ],
                "real_space_c": [
                    -0.9069119268647946,
                    -66.87864415805207,
                    -18.11622189679684,
                ],
                "space_group_hall_symbol": "P 1",
            },
            "fraction_indexed": 0.9826456729111597,
            "indexed_score": 0.0,
            "num_indexed": 94050.0,
            "rmsd_score": 0.022281469949387844,
            "rmsdxy": 0.0742897529006927,
            "score": 0.02283037493359119,
            "volume_score": 0.0005489049842033467,
        },
        "4": {
            "crystal": {
                "__id__": "crystal",
                "real_space_a": [
                    34.18115752153328,
                    -7.465156307918587,
                    20.502294710308526,
                ],
                "real_space_b": [
                    16.651208974058097,
                    8.074467824219235,
                    -36.09179697759565,
                ],
                "real_space_c": [
                    0.9098731780357134,
                    66.88194745238343,
                    18.117283469195094,
                ],
                "space_group_hall_symbol": "P 1",
            },
            "fraction_indexed": 0.9777141603368474,
            "indexed_score": 0.007258548785185975,
            "num_indexed": 93578.0,
            "rmsd_score": 0.0340721436612923,
            "rmsdxy": 0.0748993864764184,
            "score": 0.04210513968982979,
            "volume_score": 0.000774447243351517,
        },
    }
    assert candidate_crystals == expected_crystals_output

    with h5py.File(tmp_path / "indexed.refl") as f:
        flags = f["/dials/processing/group_0/flags"]
        assert len(flags) == 107999
        n_indexed = np.sum(np.array(flags, dtype=int) == 36)
        n_unindexed = np.sum(np.array(flags, dtype=int) == 32)
        assert n_indexed == 107265
        assert n_unindexed == 734
