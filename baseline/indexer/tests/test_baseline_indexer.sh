#!/bin/bash

# Run this file from within the build folder i.e.
# ../baseline_indexer/test_baseline_indexer.sh 

# To run the indexer, we need:
#   a strong reflection table (in DIALS new H5 format)
#   an experiment list (standard dials format)
#   a max cell (i.e. upper limit on basis vector length)

if test -f "candidate_vectors.json"; then
    rm candidate_vectors.json
fi
if test -f "candidate_crystals.json"; then
    rm candidate_crystals.json
fi

# Note This is a 'bad' datasets and does not have physically meaningful solutions
./bin/baseline_indexer \
  -r /dls/i03/data/2024/cm37235-2/processing/JBE/ins_14_24_rot/strong.refl \
  -e /dls/i03/data/2024/cm37235-2/processing/JBE/ins_14_24_rot/imported.expt \
  --max-cell 100 \
  --dmin 1.81 \
  --max-refine 5 \
  --test

output=$(cat candidate_vectors.json)

crystals_output=$(cat candidate_crystals.json)

expected_output='{
    "00": [
        -0.38288460328028456,
        -16.046345646564774,
        2.9586537526204055
    ],
    "01": [
        -41.00043348644091,
        6.374347624571426,
        53.04086788840915
    ],
    "02": [
        -25.233528614044186,
        -61.114115715026855,
        14.959117174148561
    ],
    "03": [
        -15.894061997532845,
        63.236873000860214,
        38.00999879837036
    ],
    "04": [
        -0.45249998569488525,
        -13.574999570846558,
        8.144999742507935
    ],
    "05": [
        -18.778749406337738,
        -87.10624724626541,
        -90.95249712467194
    ],
    "06": [
        -75.11499762535095,
        -18.09999942779541,
        0.0
    ],
    "07": [
        -8.144999742507935,
        -90.04749715328217,
        -25.339999198913574
    ],
    "08": [
        -59.72999811172485,
        -82.35499739646912,
        -37.33124881982803
    ],
    "09": [
        0.0,
        1.809999942779541,
        -9.954999685287476
    ],
    "10": [
        1.809999942779541,
        26.244999170303345,
        36.19999885559082
    ]
}'

expected_crystals_output='{
    "0": {
        "crystal": {
            "__id__": "crystal",
            "real_space_a": [
                -0.06961538241460075,
                2.4713460757182153,
                5.1863459898875295
            ],
            "real_space_b": [
                -0.4524999856948853,
                -13.574999570846558,
                8.144999742507936
            ],
            "real_space_c": [
                23.632374818508445,
                -0.5999207955140279,
                2.061843826220592
            ],
            "space_group_hall_symbol": "P 1"
        },
        "fraction_indexed": 0.10682492581602374,
        "indexed_score": 0.6629650127224291,
        "num_indexed": 36.0,
        "rmsd_score": 2.242712271537948,
        "rmsdxy": 4.017888157053147,
        "score": 2.9056772842603773,
        "volume_score": 0.0
    },
    "1": {
        "crystal": {
            "__id__": "crystal",
            "real_space_a": [
                0.38288460328028473,
                16.046345646564777,
                -2.9586537526204064
            ],
            "real_space_b": [
                -23.701990200923056,
                3.071266871232246,
                3.1245021636669383
            ],
            "real_space_c": [
                6.659274393549348,
                12.026569189933666,
                43.76145789256466
            ],
            "space_group_hall_symbol": "P 1"
        },
        "fraction_indexed": 0.13353115727002968,
        "indexed_score": 0.34103691783506696,
        "num_indexed": 45.0,
        "rmsd_score": 0.9626854408929544,
        "rmsdxy": 1.6545175269493617,
        "score": 4.366235416423868,
        "volume_score": 3.0625130576958473
    },
    "2": {
        "crystal": {
            "__id__": "crystal",
            "real_space_a": [
                0.38288460328028445,
                16.04634564656477,
                -2.9586537526204046
            ],
            "real_space_b": [
                41.000433486440905,
                -6.374347624571417,
                -53.04086788840913
            ],
            "real_space_c": [
                -57.86475987637722,
                -0.5001713888701264,
                -52.70489799936481
            ],
            "space_group_hall_symbol": "P 1"
        },
        "fraction_indexed": 0.16913946587537093,
        "indexed_score": 0.0,
        "num_indexed": 57.0,
        "rmsd_score": 0.0,
        "rmsdxy": 0.8489344897612263,
        "score": 5.311816511184903,
        "volume_score": 5.311816511184903
    },
    "3": {
        "crystal": {
            "__id__": "crystal",
            "real_space_a": [
                -0.452499985694886,
                -13.574999570846561,
                8.144999742507938
            ],
            "real_space_b": [
                -23.423528671264652,
                -6.814117431640623,
                -17.620881795883182
            ],
            "real_space_c": [
                40.54793350074604,
                -19.949347195417985,
                -44.89586814590123
            ],
            "space_group_hall_symbol": "P 1"
        },
        "fraction_indexed": 0.08902077151335312,
        "indexed_score": 0.9259994185562226,
        "num_indexed": 30.0,
        "rmsd_score": 0.6620759763720472,
        "rmsdxy": 1.3433182232729337,
        "score": 5.396590621478495,
        "volume_score": 3.8085152265502256
    },
    "4": {
        "crystal": {
            "__id__": "crystal",
            "real_space_a": [
                -0.4524999856948851,
                -13.574999570846556,
                8.144999742507933
            ],
            "real_space_b": [
                -23.29637154612852,
                -2.5625270929025556,
                -17.549129879992943
            ],
            "real_space_c": [
                40.54793350074602,
                -19.949347195417978,
                -44.895868145901204
            ],
            "space_group_hall_symbol": "P 1"
        },
        "fraction_indexed": 0.0830860534124629,
        "indexed_score": 1.0255350921071376,
        "num_indexed": 28.0,
        "rmsd_score": 0.7008434468768414,
        "rmsdxy": 1.3799046490920763,
        "score": 5.463242647924283,
        "volume_score": 3.7368641089403045
    }
}'

# Compare the output with the expected output
if [[ "$output" == "$expected_output" &&  "$crystals_output" == "$expected_crystals_output" ]]; then
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