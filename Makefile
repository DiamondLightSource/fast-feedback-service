
TMPDIR=/scratch/mep23677/tmp

all:

sif:
	podman build -t ffs .
	rm ffs.tar
	podman save --format docker-archive ffs -o ffs.tar
	apptainer build ffs.sif docker-archive://ffs.tar
