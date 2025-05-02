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
# |  40.71  40.75  69.58  92.05  91.95  98.10 |   114112  0.00 |   86754   98  0.01 |   0.72     0.00 |          0.02 |
# |  40.70  40.82  69.58  92.06  91.94  98.10 |   114282  0.01 |   86821   98  0.01 |   0.74     0.02 |          0.04 |
# |  40.70  40.82  69.65  92.02  91.98  98.10 |   114405  0.01 |   87561   99  0.00 |   0.75     0.04 |          0.05 |
# |  40.54  40.77  69.58  91.73  92.12  97.65 |   113836  0.00 |   82599   94  0.08 |   0.77     0.09 |          0.18 |
# |  40.54  40.85  69.58  92.23  92.12  98.32 |   113811  0.00 |   86359   98  0.02 |   0.86     0.24 |          0.26 |

./bin/baseline_indexer \
  -r /dls/i03/data/2024/cm37235-2/processing/JBE/c2sum/strong.refl \
  -e /dls/i03/data/2024/cm37235-2/processing/JBE/c2sum/imported.expt \
  --max-cell 94.4 \
  --dmin 1.94 \
  --max-refine 5 \
  --test

crystals_output=$(cat candidate_crystals.json)

expected_crystals_output='{
    "0": {
        "crystal": {
            "__id__": "crystal",
            "real_space_a": [
                34.33800101280212,
                -7.4536844303733405,
                20.56400060653686
            ],
            "real_space_b": [
                16.684000492095947,
                8.117368660475076,
                -36.278001070022576
            ],
            "real_space_c": [
                0.862222247653536,
                67.14555753601923,
                18.214444981680966
            ],
            "space_group_hall_symbol": "P 1"
        },
        "fraction_indexed": 0.9834827855936334,
        "indexed_score": 0.013358152950987922,
        "num_indexed": 86754.0,
        "rmsd_score": 0.0,
        "rmsdxy": 0.7247468497481094,
        "score": 0.017175548983899013,
        "volume_score": 0.003817396032911091
    },
    "1": {
        "crystal": {
            "__id__": "crystal",
            "real_space_a": [
                34.30736943295129,
                -7.453684430373345,
                20.591228677515403
            ],
            "real_space_b": [
                16.745263651797632,
                8.117368660475083,
                -36.33245721197965
            ],
            "real_space_c": [
                0.8622222476535424,
                67.14555753601923,
                18.214444981680973
            ],
            "space_group_hall_symbol": "P 1"
        },
        "fraction_indexed": 0.9842423280543243,
        "indexed_score": 0.012244391492940945,
        "num_indexed": 86821.0,
        "rmsd_score": 0.023070712289572137,
        "rmsdxy": 0.7364297297577785,
        "score": 0.04127994180261492,
        "volume_score": 0.005964838020101837
    },
    "2": {
        "crystal": {
            "__id__": "crystal",
            "real_space_a": [
                34.307369432951276,
                -7.453684430373343,
                20.591228677515392
            ],
            "real_space_b": [
                16.745263651797636,
                8.117368660475073,
                -36.33245721197964
            ],
            "real_space_c": [
                0.8423684458983591,
                67.23631777261433,
                18.16622860598982
            ],
            "space_group_hall_symbol": "P 1"
        },
        "fraction_indexed": 0.9926313044858351,
        "indexed_score": 0.0,
        "num_indexed": 87561.0,
        "rmsd_score": 0.04058422607473455,
        "rmsdxy": 0.7454240588830259,
        "score": 0.04809997670970356,
        "volume_score": 0.0075157506349690095
    },
    "3": {
        "crystal": {
            "__id__": "crystal",
            "real_space_a": [
                34.16555656327142,
                -7.544444666968449,
                20.47777838177152
            ],
            "real_space_b": [
                17.028889391157357,
                8.298889133665291,
                -36.10555662049188
            ],
            "real_space_c": [
                0.8622222476535404,
                67.14555753601921,
                18.214444981680973
            ],
            "space_group_hall_symbol": "P 1"
        },
        "fraction_indexed": 0.936379816576164,
        "indexed_score": 0.08416411557248972,
        "num_indexed": 82599.0,
        "rmsd_score": 0.09145492528933774,
        "rmsdxy": 0.7721772674623144,
        "score": 0.17593681124688,
        "volume_score": 0.0003177703850525404
    },
    "4": {
        "crystal": {
            "__id__": "crystal",
            "real_space_a": [
                34.16555656327142,
                -7.544444666968447,
                20.477778381771515
            ],
            "real_space_b": [
                16.603450782117783,
                8.026608423879974,
                -36.44590750772354
            ],
            "real_space_c": [
                0.8622222476535354,
                67.14555753601923,
                18.214444981680973
            ],
            "space_group_hall_symbol": "P 1"
        },
        "fraction_indexed": 0.9790048860119486,
        "indexed_score": 0.01994189339946945,
        "num_indexed": 86359.0,
        "rmsd_score": 0.24152461201564218,
        "rmsdxy": 0.8568257093888961,
        "score": 0.2614665054151116,
        "volume_score": 0.0
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