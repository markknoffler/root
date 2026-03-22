# TMVA SOFIE Parser Regression Tests

Python regression tests for SOFIE parsers. These tests are intended to reproduce
known bugs and verify fixes.

## Requirements

- ROOT built with PyROOT (tpython=ON)
- Keras (for Keras parser tests)

## Running the tests

From the ROOT source or build directory, with ROOT's Python environment:

```bash
python tmva/sofie/tests/test_keras_parser_global_average_pool.py -v
```

Or using unittest discovery:

```bash
python -m unittest tmva.sofie.tests.test_keras_parser_global_average_pool -v
```

## Test: GlobalAveragePool typo (test_keras_parser_global_average_pool.py)

Reproduces the `GloabalAveragePool` typo bug in `pooling.py` line 64. When the
typo exists, parsing a model with GlobalAveragePooling2D raises `AttributeError`.
The test passes when the correct enum `GlobalAveragePool` is used.
