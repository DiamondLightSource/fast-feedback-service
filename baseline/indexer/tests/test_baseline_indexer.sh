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
if test -f "indexed.refl"; then
    rm indexed.refl
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
                -0.19426335071900225,
                2.7801174067467365,
                6.051001054208426
            ],
            "real_space_b": [
                0.08619117365755113,
                -15.44256627332296,
                8.780280738821919
            ],
            "real_space_c": [
                26.776181405061205,
                -0.4458485817621857,
                2.390114898090674
            ],
            "space_group_hall_symbol": "P 1"
        },
        "fraction_indexed": 0.10682492581602374,
        "indexed_score": 0.6629650127224291,
        "num_indexed": 36.0,
        "rmsd_score": 2.2201939081181536,
        "rmsdxy": 3.380438094882832,
        "score": 2.883158920840583,
        "volume_score": 0.0
    },
    "1": {
        "crystal": {
            "__id__": "crystal",
            "real_space_a": [
                0.4481316974199212,
                15.415927201656164,
                -3.0275331516335937
            ],
            "real_space_b": [
                -23.669999591400334,
                2.451655058915848,
                3.0805873818471756
            ],
            "real_space_c": [
                6.859148012753957,
                10.20967906616398,
                43.357950357525134
            ],
            "space_group_hall_symbol": "P 1"
        },
        "fraction_indexed": 0.13353115727002968,
        "indexed_score": 0.34103691783506696,
        "num_indexed": 45.0,
        "rmsd_score": 0.862557286804198,
        "rmsdxy": 1.3191179831556559,
        "score": 3.6274071965532717,
        "volume_score": 2.423812991914007
    },
    "2": {
        "crystal": {
            "__id__": "crystal",
            "real_space_a": [
                -0.4068140301237836,
                -14.2335896432367,
                8.564555114263998
            ],
            "real_space_b": [
                -22.948009311618886,
                -7.172171064419543,
                -16.887995768461504
            ],
            "real_space_c": [
                40.19545822862181,
                -20.374384894187973,
                -43.00228376073281
            ],
            "space_group_hall_symbol": "P 1"
        },
        "fraction_indexed": 0.08902077151335312,
        "indexed_score": 0.9259994185562226,
        "num_indexed": 30.0,
        "rmsd_score": 0.5276192295238264,
        "rmsdxy": 1.0458207131523087,
        "score": 4.709400506327247,
        "volume_score": 3.2557818582471985
    },
    "3": {
        "crystal": {
            "__id__": "crystal",
            "real_space_a": [
                0.37266400808892014,
                16.230354909114038,
                -3.033100126735198
            ],
            "real_space_b": [
                40.74989194614894,
                -6.681288152933564,
                -52.74916747787913
            ],
            "real_space_c": [
                -57.517040647405615,
                -1.0771626705005324,
                -52.54106573437347
            ],
            "space_group_hall_symbol": "P 1"
        },
        "fraction_indexed": 0.16913946587537093,
        "indexed_score": 0.0,
        "num_indexed": 57.0,
        "rmsd_score": 0.0,
        "rmsdxy": 0.7254843104456131,
        "score": 4.755675887505278,
        "volume_score": 4.755675887505278
    },
    "4": {
        "crystal": {
            "__id__": "crystal",
            "real_space_a": [
                -0.414850669348278,
                -13.79394632605264,
                8.213075956315434
            ],
            "real_space_b": [
                -23.24216930086044,
                -2.795651431456891,
                -17.425860500421777
            ],
            "real_space_c": [
                40.40890491894017,
                -19.62342856433785,
                -44.5481398169253
            ],
            "space_group_hall_symbol": "P 1"
        },
        "fraction_indexed": 0.0830860534124629,
        "indexed_score": 1.0255350921071376,
        "num_indexed": 28.0,
        "rmsd_score": 0.7329675955857827,
        "rmsdxy": 1.205794384007887,
        "score": 4.943956407128037,
        "volume_score": 3.1854537194351167
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