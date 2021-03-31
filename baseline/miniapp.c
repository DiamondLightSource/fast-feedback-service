#include <stdio.h>
#include <stdlib.h>
#include "miniapp.h"

int main(int argc, char **argv) {
    if (argc == 2) {
        fprintf(stderr, "%s foobar.nxs foobar_000001.h5\n", argv[0]);
        return 1;
    }

    if (setup_hdf5_files(argv[1], argv[2]) < 0) {
      fprintf(stderr, "<shrug> bad thing </shrug>\n");
      exit(1);
    }

    return 0;
}
