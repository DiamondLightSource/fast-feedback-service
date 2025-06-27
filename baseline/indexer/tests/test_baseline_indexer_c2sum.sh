#!/bin/bash

# Run this file from within the build folder i.e.
# ../baseline_indexer/test_baseline_indexer_c2sum.sh 

# To run the indexer, we need:
#   a strong reflection table (in DIALS new H5 format)
#   an experiment list (standard dials format)
#   a max cell (i.e. upper limit on basis vector length)

if test -f "candidate_vectors.json"; then # Note we don't use this in this test (there are 80 vectors).
    rm candidate_vectors.json
fi
if test -f "candidate_crystals.json"; then
    rm candidate_crystals.json
fi

# This dataset in the c2sum beta lactamase dataset (https://zenodo.org/records/1014387)
# It is useful for testing that the non-primitive-basis correction is applied in indexing.
# The output here is identical to dials.index (when candidate refinement is turned off).
# The output table in the log should read:
# | Unit cell                                 | volume & score | #indexed % & score | rmsd_xy & score | overall score |
# |  40.55  40.55  69.29  92.01  91.97  98.08 |   112642  0.00 |   93724   98  0.01 |   0.07     0.00 |          0.01 |
# |  40.56  40.56  69.30  92.01  91.97  98.08 |   112717  0.00 |   94009   98  0.00 |   0.07     0.01 |          0.01 |
# |  40.54  40.55  69.29  92.02  91.97  98.08 |   112629  0.00 |   93398   98  0.01 |   0.07     0.01 |          0.02 |
# |  40.56  40.55  69.29  91.97  92.02  98.08 |   112671  0.00 |   94050   98  0.00 |   0.07     0.02 |          0.02 |
# |  40.55  40.56  69.30  92.01  91.97  98.08 |   112689  0.00 |   93578   98  0.01 |   0.07     0.03 |          0.04 |

./bin/baseline_indexer \
  -r /dls/i03/data/2024/cm37235-2/processing/JBE/c2sum/strong.refl \
  -e /dls/i03/data/2024/cm37235-2/processing/JBE/c2sum/imported.expt \
  --max-cell 94.4 \
  --dmin 1.84 \
  --max-refine 5 \
  --test

crystals_output=$(cat candidate_crystals.json)

expected_crystals_output='{
    "0": {
        "crystal": {
            "__id__": "crystal",
            "real_space_a": [
                34.17640573400373,
                -7.465908236923297,
                20.499161034425466
            ],
            "real_space_b": [
                16.648886142399864,
                8.075218184328115,
                -36.086479295366246
            ],
            "real_space_c": [
                0.90929797277945,
                66.87063326644437,
                18.121049774287396
            ],
            "space_group_hall_symbol": "P 1"
        },
        "fraction_indexed": 0.9792395858365287,
        "indexed_score": 0.005009416169085765,
        "num_indexed": 93724.0,
        "rmsd_score": 0.0,
        "rmsdxy": 0.07315121148177929,
        "score": 0.005182182229071939,
        "volume_score": 0.00017276605998617356
    },
    "1": {
        "crystal": {
            "__id__": "crystal",
            "real_space_a": [
                34.183737336394486,
                -7.466708160824077,
                20.504737658391274
            ],
            "real_space_b": [
                16.651981990166604,
                8.074837535667207,
                -36.094844734749685
            ],
            "real_space_c": [
                0.9088268074740147,
                66.88708059063406,
                18.120768250267105
            ],
            "space_group_hall_symbol": "P 1"
        },
        "fraction_indexed": 0.9822172999968656,
        "indexed_score": 0.0006290631941424409,
        "num_indexed": 94009.0,
        "rmsd_score": 0.010656318685108968,
        "rmsdxy": 0.07369353583507099,
        "score": 0.01241280945645382,
        "volume_score": 0.0011274275772024112
    },
    "2": {
        "crystal": {
            "__id__": "crystal",
            "real_space_a": [
                34.174144922093575,
                -7.465314578907954,
                20.498111897894855
            ],
            "real_space_b": [
                16.649639413574143,
                8.073981992730776,
                -36.08565154324145
            ],
            "real_space_c": [
                0.9062186699312748,
                66.86731447368523,
                18.122817840626766
            ],
            "space_group_hall_symbol": "P 1"
        },
        "fraction_indexed": 0.9758334987618978,
        "indexed_score": 0.010036286971724262,
        "num_indexed": 93398.0,
        "rmsd_score": 0.012429125731538093,
        "rmsdxy": 0.07378414730719672,
        "score": 0.022465412703262355,
        "volume_score": 0.0
    },
    "3": {
        "crystal": {
            "__id__": "crystal",
            "real_space_a": [
                -16.649288830861,
                -8.071480003169183,
                36.09015283684615
            ],
            "real_space_b": [
                -34.18080008840855,
                7.4627986369941794,
                -20.50092030704733
            ],
            "real_space_c": [
                -0.9069119268647946,
                -66.87864415805207,
                -18.11622189679684
            ],
            "space_group_hall_symbol": "P 1"
        },
        "fraction_indexed": 0.9826456729111597,
        "indexed_score": 0.0,
        "num_indexed": 94050.0,
        "rmsd_score": 0.022281469949387844,
        "rmsdxy": 0.0742897529006927,
        "score": 0.02283037493359119,
        "volume_score": 0.0005489049842033467
    },
    "4": {
        "crystal": {
            "__id__": "crystal",
            "real_space_a": [
                34.18115752153328,
                -7.465156307918587,
                20.502294710308526
            ],
            "real_space_b": [
                16.651208974058097,
                8.074467824219235,
                -36.09179697759565
            ],
            "real_space_c": [
                0.9098731780357134,
                66.88194745238343,
                18.117283469195094
            ],
            "space_group_hall_symbol": "P 1"
        },
        "fraction_indexed": 0.9777141603368474,
        "indexed_score": 0.007258548785185975,
        "num_indexed": 93578.0,
        "rmsd_score": 0.0340721436612923,
        "rmsdxy": 0.0748993864764184,
        "score": 0.04210513968982979,
        "volume_score": 0.000774447243351517
    }
}'

# Compare the output with the expected output
if [ "$crystals_output" == "$expected_crystals_output" ]; then
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