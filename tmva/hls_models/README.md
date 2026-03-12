# HLS4ML Parser for TMVA SOFIE

This document describes the HLS4ML parser implementation for TMVA SOFIE (System for Optimized Fast Inference Engine). The parser enables parsing of HLS4ML `ModelGraph` objects into SOFIE's internal representation, supporting machine learning inference workflows that bridge HLS4ML (FPGA-oriented) and SOFIE (C++ inference).

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Project Structure](#project-structure)
4. [HLS4ML Exploration](#hls4ml-exploration)
5. [Parsing Function: Design and Algorithm](#parsing-function-design-and-algorithm)
6. [Model Configuration Schema](#model-configuration-schema)
7. [SOFIE RModel Integration](#sofie-rmodel-integration)
8. [Supported Operators](#supported-operators)
9. [Weight Handling](#weight-handling)
10. [Tests](#tests)
11. [Running the Tests](#running-the-tests)
12. [Implementation Notes and Resolved Issues](#implementation-notes-and-resolved-issues)
13. [References](#references)

---

## Overview

The implementation consists of three main components:

1. **HLS4ML exploration** — Understanding the HLS4ML library, its `ModelGraph` API, layer structure, and conversion pipelines (Keras → HLS4ML, ONNX → HLS4ML).

2. **Configuration extraction** — A parsing function that takes an in-memory HLS4ML `ModelGraph` and returns a structured model configuration dictionary suitable for downstream processing.

3. **SOFIE RModel construction** — Python bindings that consume the parsed configuration to instantiate a SOFIE `RModel` via PyROOT, supporting ReLU, Elu, Gemm, Reshape, and Concat operators.

The design follows the architecture of the existing Keras parser in TMVA, with a modular layer-by-layer construction approach.

---

## Prerequisites

- **ROOT** with TMVA and SOFIE enabled
- **Python 3.8+**
- **hls4ml** — HLS4ML package for ML inference on FPGAs
- **Keras** / **TensorFlow** — For Keras model conversion to HLS4ML
- **ONNX** and **qonnx** (optional) — For ONNX model conversion to HLS4ML

---

## Project Structure

```
tmva/hls_models/
├── README.md
├── ConvWithAsymmetricPadding.onnx    # Example ONNX model for testing
└── hls4ml_parser/
    ├── __init__.py
    ├── config.py                     # extract_hls_config() — parsing function
    ├── parser.py                     # PyHLS4ML.ParseFromModelGraph() — RModel builder
    └── layers/
        ├── relu.py                   # MakeHLSReLU
        ├── elu.py                    # MakeHLSELU
        ├── gemm.py                   # MakeHLSGemm (Dense)
        ├── reshape.py                # MakeHLSReshape / Flatten
        └── concat.py                 # MakeHLSConcat

tmva/sofie/exercise_outputs/
├── exercise4_hls4ml_parser_test.py   # Test for parsing (config extraction)
├── exercise5_hls4ml_rmodel_test.py   # Test for RModel construction
├── exercise4_hls4ml_modelgraph_output.txt
├── exercise4_hls4ml_config.json
├── HLS4MLDenseModel_sofie.hxx        # Generated C++ header
└── HLS4MLDenseModel_sofie.dat        # Generated weight file
```

---

## HLS4ML Exploration

HLS4ML is a Python library for deploying machine learning models on FPGAs via High-Level Synthesis (HLS). Key concepts:

- **ModelGraph** — The core in-memory representation of a model after conversion from Keras, ONNX, or other frameworks. It exposes:
  - `get_layers()` / `layers` — list of layer objects
  - `inputs` / `outputs` — model-level input and output variables
  - `name` — model identifier

- **Layer structure** — Each layer typically provides:
  - `name`, `inputs`, `outputs` — topology
  - `attributes` — layer-specific parameters (e.g., activation type, axis)
  - `get_output_variable()` — output shape information
  - `get_weights()` — weight tensors (where applicable, e.g., Dense layers)

- **Conversion pipelines**:
  - Keras → HLS4ML: `hls4ml.converters.convert_from_keras_model()`
  - ONNX → HLS4ML: `hls4ml.converters.convert_from_onnx_model()`

Understanding this structure informed the parsing algorithm and configuration schema described below.

---

## Parsing Function: Design and Algorithm

The parsing function `extract_hls_config(hls_model)` is implemented in `config.py`. It is the central piece for configuration extraction and is designed for clarity, robustness, and compatibility with various HLS4ML model structures.

### Algorithm

1. **Model-level metadata**
   - Read model `name` from `hls_model`; use `"hls_model"` as default if missing.

2. **Layer traversal**
   - Obtain the layer sequence via `get_layers()` if available, otherwise `layers`.
   - For each layer, extract:
     - `name` — layer identifier
     - `class_name` — Python class name (e.g., `Dense`, `Activation`)
     - `attributes` — dictionary of layer attributes from `layer.attributes`
     - `inputs` — names of input variables
     - `outputs` — names of output variables
     - `output_shape` — shape from `layer.get_output_variable().shape` when available

3. **Weight metadata**
   - For layers with `get_weights()`, collect weight names and shapes.
   - Keys use the form `layer_name/weight_name`; values store shape information for later use.

4. **Model-level inputs and outputs**
   - Aggregate input and output names from `hls_model.inputs` and `hls_model.outputs`.

5. **Robustness**
   - Use `getattr` and try/except to handle missing attributes or non-standard layer structures.
   - Graceful fallbacks avoid failures on optional fields.

### Implementation location

- File: `tmva/hls_models/hls4ml_parser/config.py`
- Entry point: `extract_hls_config(hls_model)`

---

## Model Configuration Schema

The function returns a Python dictionary with the following structure:

```python
{
    "name": str,              # Model name
    "layers": [               # Ordered list of layer configurations
        {
            "name": str,
            "class_name": str,      # e.g., "Dense", "Activation"
            "attributes": dict,     # Layer-specific attributes
            "inputs": list,         # Input tensor names
            "outputs": list,        # Output tensor names
            "output_shape": list | None  # Output shape when available
        },
        ...
    ],
    "weights": {              # Weight metadata: layer_name/weight_name -> shape
        "dense/weight": {"shape": [out, in]},
        "dense/bias": {"shape": [out]},
        ...
    },
    "inputs": list,           # Model input tensor names
    "outputs": list           # Model output tensor names
}
```

This schema captures topology, shapes, and weight metadata in a framework-agnostic form, suitable for building SOFIE models or other backends.

---

## SOFIE RModel Integration

The parser integrates with SOFIE via the PyROOT Python interface. The flow is:

1. **Parse** — `extract_hls_config(hls_model)` produces the configuration.
2. **Build** — `PyHLS4ML.ParseFromModelGraph(hls_model, name=..., keras_model=...)` constructs the `RModel`:
   - Creates an empty `SOFIE.RModel.RModel()`
   - Adds input tensor info
   - Iterates over layers, converting each to a SOFIE operator
   - Adds initialized tensors for weights
   - Sets output tensor names
3. **Generate** — The user calls `rmodel.Generate()` and `rmodel.OutputGenerated(filename)` to emit C++ code and weight files.

The architecture mirrors the Keras parser: a layer mapping table (`mapHLS4MLLayer`), per-layer factory functions (`MakeHLS*`), and an `add_layer_into_RModel()` routine that appends operators to the `RModel`.

---

## Supported Operators

| HLS4ML Layer / Activation | SOFIE Operator | File    |
|---------------------------|----------------|---------|
| Dense                     | ROperator_Gemm | gemm.py |
| ReLU / relu               | ROperator_Relu | relu.py |
| ELU / elu                 | ROperator_Elu  | elu.py  |
| Reshape                   | ROperator_Reshape | reshape.py |
| Flatten                   | ROperator_Reshape | reshape.py |
| Concatenate / Concat      | ROperator_Concat | concat.py |

- **Gemm** — Dense layers are represented as `Y = X @ W + B` with `transA=0`, `transB=0` to match Keras weight layout `[in, out]`.
- **Activation** — ReLU and ELU are supported both as standalone layers and as activations on Dense layers.
- **Reshape / Flatten** — Shape tensors are added via `AddInitializedTensor`; Flatten uses `target_shape = [-1]`.

---

## Weight Handling

- **Source** — Weights are obtained from HLS4ML layer objects via `get_weights()` or `weights`. For models converted from Keras, `keras_model` can be passed so weights are read from the original Keras model when HLS4ML does not expose them.
- **Storage** — Weights are passed to SOFIE as contiguous NumPy arrays via `AddInitializedTensor`. Arrays are flattened and cast to `float32` for Gemm; shape tensors use `int64`.
- **Transpose** — Gemm uses `transB=0` to align with Keras Dense weight layout `[in_features, out_features]`.

---

## Tests

Two test scripts exercise the implementation:

### 1. `exercise4_hls4ml_parser_test.py`

- **Purpose**: Validates the parsing function `extract_hls_config()`.
- **Input**: ONNX model (`ConvWithAsymmetricPadding.onnx`) converted to HLS4ML.
- **Actions**:
  - Converts ONNX to HLS4ML (with qonnx cleanup and channels-last conversion).
  - Calls `extract_hls_config(hls_model)`.
  - Writes a text summary and JSON configuration to `exercise_outputs/`.

**Outputs**:
- `exercise4_hls4ml_modelgraph_output.txt` — Human-readable summary
- `exercise4_hls4ml_config.json` — Serialized configuration (JSON-safe)

### 2. `exercise5_hls4ml_rmodel_test.py`

- **Purpose**: Validates full parsing and RModel construction for supported operators.
- **Input**: Keras Sequential model (Dense 16 + ReLU, Dense 8 + ELU, Dense 4).
- **Actions**:
  - Builds the Keras model and converts it to HLS4ML.
  - Calls `PyHLS4ML.ParseFromModelGraph(hls_model, keras_model=model)`.
  - Calls `rmodel.Generate()` and `rmodel.OutputGenerated()`.
  - Produces C++ header and weight file.

**Outputs**:
- `HLS4MLDenseModel_sofie.hxx` — Generated inference header
- `HLS4MLDenseModel_sofie.dat` — Weight file

---

## Running the Tests

### Environment

```bash
# Activate ROOT (adjust path to your ROOT installation)
source /path/to/root-build/bin/thisroot.sh

# Use a Python environment with hls4ml, Keras, etc.
conda activate hls4ml  # or your preferred environment
```

### Run Parsing Test (Exercise 4)

```bash
cd /path/to/root
python tmva/sofie/exercise_outputs/exercise4_hls4ml_parser_test.py
```

Expected: Creation of `exercise4_hls4ml_modelgraph_output.txt` and `exercise4_hls4ml_config.json` in `tmva/sofie/exercise_outputs/`.

### Run RModel Test (Exercise 5)

```bash
cd /path/to/root
python tmva/sofie/exercise_outputs/exercise5_hls4ml_rmodel_test.py
```

Expected: Printout of generated paths and creation of `HLS4MLDenseModel_sofie.hxx` and `HLS4MLDenseModel_sofie.dat`.

### Dependencies

- **Exercise 4**: `hls4ml`, `qonnx`, `onnx`
- **Exercise 5**: `ROOT` (with `libROOTTMVASofie`), `hls4ml`, `keras`

---

## Implementation Notes and Resolved Issues

### NumPy array handling for SOFIE

Passing NumPy buffer pointers (e.g. `.data`) to SOFIE via CPyCppyy could cause element-size mismatches. The solution is to pass contiguous arrays directly:

- Use `np.ascontiguousarray(arr.flatten(), dtype="float32")` for weight tensors.
- Use contiguous `int64` arrays for shape tensors.
- Avoid using `.data`; pass the array object so the C++ bindings receive a correct buffer.

### Gemm transpose for Dense layers

Keras Dense weights use layout `[in_features, out_features]` for `Y = X @ W`. Setting `transB=0` in Gemm ensures correct matrix multiplication; `transB=1` leads to shape mismatches and broadcast errors.

### OutputGenerated behavior

`OutputGenerated(filename)` both emits the C++ header and writes the weights to a `.dat` file (by replacing `.hxx` with `.dat` in the path). No separate call for writing the weight file is required.

### Redundant files

A top-level `hls4ml_parser.py` module was removed; the package `hls4ml_parser/` (directory with `__init__.py`) is the canonical module. Imports should use `tmva.hls_models.hls4ml_parser.config` and `tmva.hls_models.hls4ml_parser.parser` or `tmva.hls_models.hls4ml_parser` for package-level exports.

---

## References

- [HLS4ML Documentation](https://fastmachinelearning.org/hls4ml/)
- [TMVA SOFIE](https://root.cern.ch/doc/master/namespaceTMVA_1_1Experimental_1_1SOFIE.html)
- [ROOT](https://root.cern/)
