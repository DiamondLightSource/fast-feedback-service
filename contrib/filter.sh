#!/usr/bin/env bash

set -euo pipefail
set -x

if [[ ! -f insert-beginning ]]; then
    wget https://github.com/newren/git-filter-repo/blob/ac50405044073df5caca22f8ac9d7a248b11515b/contrib/filter-repo-demos/insert-beginning
    chmod +x insert-beginning
    ln -s $(which git-filter-repo) git_filter_repo.py
fi

rm -rf ./miniapp
git clone file:///Users/nickd/dials/miniapp/.git ffs
cd ffs

# Get rid of other stuff
git filter-repo \
    --path h5read \
    --path cmake \
    --path common \
    --path cuda/common \
    --path cuda/spotfinder \
    --path docs \
    --path .clang-format \
    --path .gitignore \
    --path .pre-commit-config.yaml \
    --path baseline

git filter-repo --invert-paths \
    --path docs/FPGA_algorithm.drawio.pdf \
    --path 'cuda/spotfinder/watch_dc.py' \
    --path 'cuda/spotfinder/watch_shm.py'
git filter-repo \
    --path-rename cuda/spotfinder:spotfinder \
    --path-rename spotfinder/pyproject.toml:pyproject.toml \
    --path-rename spotfinder/poetry.lock:poetry.lock \
    --path-rename spotfinder/tests:tests \
    --path-rename spotfinder/service.py:src/ffs/service.py \
    --path-rename spotfinder/compare_service.py:src/ffs/compare_service.py \
    --path-rename common/include/common.hpp:include/common.hpp \
    --path-rename common/include/common.hpp:include/common.hpp \
    --path-rename cuda/common/include/common.hpp:include/cuda_common.hpp

mkdir contrib
cp ../filter.sh contrib
git add contrib
git commit -nm "Add repo-filter script, used to rewrite history"
cp ../LICENSE .
../insert-beginning --file LICENSE

# Rewrite references to issues
echo '#==>ndevenish/miniapp#' >expressions.txt
git filter-repo --replace-message expressions.txt

git remote add origin git@github.com:DiamondLightSource/fast-feedback-service.git
# git filter-repo --commit-callback "if not commit.parents: commit.file_changes.append(FileChange(b'M', '/', b'$(git hash-object -w ../LICENSE)', b'100644'))"
# --path-rename cuda/common/include/common.hpp:include/cuda_common.hpp

# git filter-repo \
#     --filename-callback "print(filename); return filename" # --path spotfinder \
