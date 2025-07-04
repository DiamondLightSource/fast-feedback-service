
TMPDIR=/scratch/mep23677/tmp

all:

sif:
	podman build -t ffs .
	rm -f ffs.tar ffs.sif
	podman save --format docker-archive ffs -o ffs.tar
	apptainer build ffs.sif docker-archive://ffs.tar
