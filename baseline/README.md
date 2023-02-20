# DIALS Baseline algorithm miniapp

This folder contains a copy of the DIALS code used for spotfinding, along with
a standalone re-implementation of the algorithm, and benchmarking and testing
code to validate that the results are the same.

## Building and Usage

If added as a subproject with CMakes [`add_subdirectory`], then a target
`standalone` will be built of the DIALS-free algorithm, with a basic C++
interface (defined in `standalone.h`). If, optionally, DIALS is found (via the
`DIALS_BUILD` environment variable pointing to the `build/` folder) then a
second `baseline_dials` target will be created with a C-API declared in
`baseline.h`.


## Targets

| Target Name      | Purpose                                                    |
| ---------------- | ---------------------------------------------------------- |
| `./bm`           | Uses Google Benchmark to run basic algorith implementations, for speed comparison.                 
| `./check_no_tbx` | Use h5read to read a nexus file or sample data, and compare the output from the original and standalone algorithm.
| `./miniapp`      | A simple miniapp for running the DIALS dispersion algorithm against a nexus file.

[Benchmark]: https://github.com/google/benchmark
[`add_subdirectory`]: https://cmake.org/cmake/help/latest/command/add_subdirectory.html