# Testing

To run the tests, you need to have pytest and dials-data available in your environment and be on a machine with an NVIDIA GPU and the CUDA toolkit installed.
(These tests assume you have built in a folder called `build`. The 32-bit
data tests use `build/bin/spotfinder32`, which the same build produces.)
Run:
```bash
python -m pytest tests/ -v --regression
```
