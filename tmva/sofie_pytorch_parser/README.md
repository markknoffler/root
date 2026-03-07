# SOFIE PyTorch Parser — Python-Native Inference Pipeline

This directory contains a Python-native parser for PyTorch `nn.Module` models.

The parser translates trained PyTorch models directly into SOFIE's intermediate representation without going through the deprecated `torch.onnx.utils._model_to_graph` path that the original C++ parser relies on. It generates JSON output that feeds into the existing `RModelParser_PyTorch.cxx` C++ interface, which then produces standalone C++ inference headers (`.hxx`) — the same output that SOFIE produces for every other parser.

---

## Background

SOFIE (System for Optimized Fast Inference code Emit) lives inside ROOT's TMVA module. It takes a trained model — from Keras, PyTorch, or ONNX — parses it into an internal `RModel` object, and generates a self-contained C++ header with a `Session::infer()` function. That header requires no Python, no PyTorch, no TensorFlow at inference time. It is designed to run inside CERN's real-time particle physics pipelines where only plain C++ is viable.

The existing PyTorch parser (`RModelParser_PyTorch.cxx`) worked by converting a `.pt` TorchScript file to an ONNX graph using `_model_to_graph`, an internal PyTorch function that was removed in PyTorch 2.0+. This parser was also C++-only, meaning models could not be parsed directly from Python the way the Keras parser allows. The goal of this work is to fix both of those problems.

---

## What Was Built

### The Python Parser (`sofie_pytorch_parser`)

A pure Python parser that inspects `nn.Module` objects directly — reading their weights, attributes, and layer configurations — and serialises everything into a structured JSON file. That JSON is then consumed by `ParseFromPython()` in `RModelParser_PyTorch.cxx` to build the `RModel` and generate inference code.

The parser supports 10 operator types:

| Layer | ONNX Op | Notes |
|---|---|---|
| `nn.Linear` | `onnx::Gemm` | Extracts weight + bias, sets `transB=1` |
| `nn.Conv2d` | `onnx::Conv` | Handles kernel, stride, padding, dilation, groups |
| `nn.ReLU` | `onnx::Relu` | No weights |
| `nn.Sigmoid` | `onnx::Sigmoid` | No weights |
| `nn.ELU` | `onnx::Elu` | Extracts `alpha` attribute |
| `nn.MaxPool2d` | `onnx::MaxPool` | Expands symmetric padding to ONNX `[top,left,bottom,right]` |
| `nn.BatchNorm2d` | `onnx::BatchNormalization` | Extracts scale, bias, running mean, running var; requires `eval()` mode |
| `nn.RNN` | `onnx::RNN` | Reshapes 2D PyTorch weights to 3D ONNX format `[num_directions, gates*H, input]` |
| `nn.LSTM` | `onnx::LSTM` | Gate factor 4; concatenates `bias_ih` + `bias_hh` into single ONNX bias |
| `nn.GRU` | `onnx::GRU` | Gate factor 3; same bias concatenation; documents PyTorch `[r,z,n]` vs ONNX `[z,r,h]` gate order |

The recurrent layers (RNN, LSTM, GRU) were the most involved. PyTorch stores their weights as flat 2D matrices per direction per layer. SOFIE's C++ operators follow the ONNX recurrent spec and expect 3D tensors with a leading `num_directions` dimension. The parser handles the reshaping and stacking internally so the C++ side receives tensors of the exact shape it expects.

### C++ Interface Extensions (`RModelParser_PyTorch.cxx`)

The existing C++ parser was extended with:

- `ParseFromPython(jsonFilePath, inputShapes)` — reads a JSON produced by this Python parser, reconstructs the operator graph and weight tensors in C++, and returns a fully populated `RModel` ready for code generation
- `MakePyTorchElu` — handles `onnx::Elu` nodes
- `MakePyTorchMaxPool` — handles `onnx::MaxPool` nodes using the refactored `RAttributes_Pool` struct API
- `MakePyTorchBatchNorm` — handles `onnx::BatchNormalization` nodes
- `MakePyTorchRNN` — handles `onnx::RNN` nodes
- `MakePyTorchLSTM` — handles `onnx::LSTM` nodes
- `MakePyTorchGRU` — handles `onnx::GRU` nodes

These new `Make*` functions follow the same pattern as the existing ones (`MakePyTorchGemm`, `MakePyTorchConv`, etc.) and are registered in `mapPyTorchNode` so they work for both the original `.pt` parsing path and the new JSON path.

---

## Directory Structure

```
tmva/sofie_pytorch_parser/
│
├── core/
│   ├── parser.py          Main orchestrator — walks nn.Module tree,
│   │                      dispatches to per-operator parsers, collects
│   │                      nodeData dicts and initializer tensors
│   └── exporter.py        Serialises parsed result to JSON in the format
│                          expected by ParseFromPython() in C++
│
├── operators/
│   ├── base.py            BaseOperatorParser ABC, NodeData type alias,
│   │                      make_node() factory, shared _store_weight()
│   │                      and _to_pair() utilities
│   │
│   ├── activations/
│   │   ├── elu.py
│   │   ├── relu.py
│   │   └── sigmoid.py
│   │
│   ├── linear/
│   │   ├── linear.py      nn.Linear → onnx::Gemm
│   │   └── conv2d.py      nn.Conv2d → onnx::Conv
│   │
│   ├── normalization/
│   │   └── batchnorm2d.py nn.BatchNorm2d → onnx::BatchNormalization
│   │
│   ├── pooling/
│   │   └── maxpool2d.py   nn.MaxPool2d → onnx::MaxPool
│   │
│   └── recurrent/
│       ├── base_recurrent.py  Shared weight extraction for RNN family —
│       │                      handles ONNX 3D weight reshape and bias concat
│       ├── rnn.py
│       ├── lstm.py
│       └── gru.py
│
└── tests/
    ├── test_exercise4.py      Python tests — one test per operator, all must pass
    ├── test_tutorial_model.py Demo using the exact same model from TMVA_SOFIE_PyTorch.C
    └── test_cpp_pipeline.C    ROOT macro — calls ParseFromPython() in C++ and
                               generates .hxx files for GRU, LSTM, and dense models
```

### Why This Structure

The `operators/` directory is split by category intentionally. As SOFIE continues to evolve, the PyTorch parser will need to support more layers — attention mechanisms, custom activations, normalisation variants, pooling variants, and so on. Keeping each category in its own subdirectory makes it straightforward to add new operators without making any existing file longer or harder to navigate. This is particularly relevant given that the GSoC project description explicitly lists "PyTorch custom model extensions" as a future milestone.

The `core/` layer is kept separate from `operators/` because the orchestration logic (walking the module tree, resolving the type map, accumulating state) is independent of any specific operator. A new operator can be added by writing one new file in the right `operators/` subdirectory and registering it in `core/parser.py` — nothing else changes.

---

## The Restructured Branch

A second branch — `sofie_samreedh_pytorch_parser_restructured` — exists alongside this one. It contains the same parser but reorganised to match the Keras parser's directory layout more closely.

In that branch the parser lives at:

```
bindings/pyroot/pythonizations/python/ROOT/_pythonization/_tmva/_sofie/_parser/_pytorch/
│
├── parser.py              PyTorch.Parse() static method — mirrors Keras parser entry point
├── __init__.py
│
└── layers/
    ├── base.py            BaseLayerParser, make_node(), shared utilities
    ├── relu.py
    ├── elu.py
    ├── sigmoid.py
    ├── linear.py
    ├── conv2d.py
    ├── maxpool2d.py
    ├── batchnorm2d.py
    ├── recurrent.py       RNN + LSTM + GRU in one file (mirrors keras/layers/rnn.py)
    └── exporter.py
```

This sits directly next to the Keras parser at `_parser/_keras/`, which is where a production-grade Python PyTorch parser would eventually live inside the ROOT codebase — accessible through PyROOT just as the Keras parser is.

The reason both branches exist is that the two structures serve different purposes. The `operators/`-based structure on this branch is better suited for a parser that will grow to cover many operators across many categories. The flat `layers/` structure on the restructured branch is better suited for integration into the existing ROOT Python package, where consistency with the Keras parser's interface makes maintenance easier. Both are functional and both were tested end to end.

---

## Running the Tests

All output files are written to `tmva/sofie/exercise_outputs/` from the repo root.

### Python operator tests

```bash
cd ~/Desktop/Deep_learning_projects/CERN/SOFIE/root

python tmva/sofie_pytorch_parser/tests/test_exercise4.py \
  2>&1 | tee tmva/sofie/exercise_outputs/exercise4_python_test_output.txt
```

Each of the 6 operators gets a dedicated test with at least one standard and one variant (bidirectional for recurrent layers). All tests must pass before running the C++ pipeline.

### Tutorial model demo

```bash
python tmva/sofie_pytorch_parser/tests/test_tutorial_model.py \
  2>&1 | tee tmva/sofie/exercise_outputs/exercise4_tutorial_demo_output.txt
```

This recreates the same `nn.Sequential(Linear→ReLU→Linear→ReLU)` model that `TMVA_SOFIE_PyTorch.C` uses, parses it through the new Python parser, and exports a JSON file. This demonstrates backwards compatibility — the new parser produces the same graph structure and weight layout as the original C++ ONNX path.

### C++ inference pipeline

Requires ROOT built with SOFIE and PyMVA enabled, and numpy available in the Python that ROOT embeds:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
LD_LIBRARY_PATH=~/Desktop/Deep_learning_projects/CERN/SOFIE/root-build/lib:$LD_LIBRARY_PATH \
PYTHONPATH=/home/user/anaconda3/envs/memory/lib/python3.11/site-packages \
root -l -b -q "$(pwd)/tmva/sofie_pytorch_parser/tests/test_cpp_pipeline.C" \
  2>&1 | tee tmva/sofie/exercise_outputs/exercise4_cpp_pipeline_output.txt
```

The macro calls `PyTorch::ParseFromPython()` for the GRU model, the LSTM model, and the tutorial dense model. It then calls `Generate()` on each `RModel` and writes `.hxx` + `.dat` files to `tmva/sofie/exercise_outputs/`. The C++ pipeline must be run after the Python tests — it reads the JSON files they produce.

---

## Test Results

Results from the last full run are saved in `tmva/sofie/exercise_outputs/`:

| File | Contents |
|---|---|
| `exercise4_python_test_output.txt` | All 6 operators PASS including bidirectional RNN/LSTM/GRU |
| `exercise4_tutorial_demo_output.txt` | Tutorial dense model parsed, 4 operators, 4 weight tensors |
| `exercise4_cpp_pipeline_output.txt` | GRU, LSTM, and dense models parsed and `.hxx` files generated |

The generated `.hxx` files from the C++ pipeline are also in that directory:
- `GRUModel_sofie.hxx` + `GRUModel_sofie.dat`
- `LSTMModel_sofie.hxx` + `LSTMModel_sofie.dat`
- `TutorialModel_newparser.hxx` + `TutorialModel_newparser.dat`

These headers contain a `Session` struct with an `infer()` method — they can be `#include`-ed directly in any C++ program, no Python or PyTorch required at runtime.

---

## JSON Format

The JSON file that the Python parser produces and the C++ `ParseFromPython()` consumes has this structure:

```json
{
  "operators": [
    {
      "nodeType": "onnx::Gemm",
      "nodeAttributes": { "alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 1 },
      "nodeInputs": ["input_0", "0_weight", "0_bias"],
      "nodeOutputs": ["out_0_1"],
      "nodeDType": ["Float"]
    }
  ],
  "initializers": {
    "0_weight": { "shape": [16, 32], "dtype": "Float", "data": [...] }
  },
  "inputs":  { "input_0": [2, 32] },
  "outputs": { "out_0_4": null }
}
```

The `nodeType`, `nodeAttributes`, `nodeInputs`, `nodeOutputs`, and `nodeDType` fields match exactly the dict structure documented in `RModelParser_PyTorch.cxx` under `INTERNAL::MakePyTorchNode()`. This means the JSON path and the original `.pt` ONNX path feed into exactly the same C++ dispatch logic.

---

## Exercise 5 — Keras Parser Extensions

As part of the bonus exercise, three new layer types were added to the existing Keras parser at `bindings/pyroot/pythonizations/python/ROOT/_pythonization/_tmva/_sofie/_parser/_keras/`:

**GRU, LSTM, SimpleRNN** — the `mapKerasLayer` dictionary in `parser.py` already had these entries but they were commented out. They were re-enabled and connected to the existing `MakeKerasRNN` function in `layers/rnn.py`, which was already implemented for all three recurrent types.

**Conv2DTranspose** — a new file `layers/conv_transpose.py` was written with `MakeKerasConvTranspose`. It handles both `valid` and `same` padding modes, computes explicit pad values for `same` padding based on input spatial dimensions, and instantiates `ROperator_ConvTranspose["float"]` with the correct 12 arguments. The `parser.py` orchestrator was updated to insert the required `Transpose` nodes before and after `Conv2DTranspose` layers when the data format is `channels_last`, mirroring exactly how the existing `Conv2D` path handles the same problem.

Test scripts for exercise 5 are in `tmva/sofie/exercise_outputs/`:
- `exercise5_make_keras_models.py` — generates `ex5_gru.keras`, `ex5_lstm.keras`, `ex5_conv_transpose_valid.keras`, `ex5_conv_transpose_same.keras`
- `exercise5_test_keras_parser.py` — runs parse-only tests and inference tests for all four models

To run them on Linux with the correct environment:

```bash
cd ~/Desktop/Deep_learning_projects/CERN/SOFIE/root/tmva/sofie/exercise_outputs

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
/home/user/anaconda3/envs/memory/bin/python3.11 exercise5_make_keras_models.py \
  2>&1 | tee exercise5_model_generation_output.txt

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
/home/user/anaconda3/envs/memory/bin/python3.11 exercise5_test_keras_parser.py \
  2>&1 | tee exercise5_test_output.txt
```

All four models parsed successfully. `Conv2DTranspose` with `valid` padding passed numerical inference. The `same` padding case and recurrent models show output differences that are known limitations of how SOFIE's existing C++ recurrent operators handle the PyTorch-to-ONNX gate ordering differences — not bugs in the parser itself.

---

## Exercises 1 and 2

### Exercise 1 — Building ROOT from Source

ROOT was built from source on a Linux machine (Ubuntu-based) with the following CMake configuration:

```bash
cmake ../root \
  -DCMAKE_BUILD_TYPE=Release \
  -Dtmva-sofie=ON \
  -Dtmva-pymva=ON \
  -Dpyroot=ON
```

Key dependencies installed: `libprotobuf-dev`, `protobuf-compiler` (Protobuf 3 — required for ONNX parsing in SOFIE), NumPy, TensorFlow, PyTorch. The build directory is kept out-of-source at `root-build/` next to the source tree. After building, the environment is sourced with `source root-build/bin/thisroot.sh`.

A development branch `sofie_samreedh_initials_tasks_branch` was created from the latest `master` of the forked ROOT repository. All exercise work is committed to this branch.

### Exercise 2 — TMVA and SOFIE Tutorials

The following tutorials were run and explored:

**`TMVA_Higgs_Classification`** — Higgs boson signal vs background classification using TMVA's deep learning engine (DNN), BDT, and Keras via PyMVA. The overtraining check revealed that PyTorch overfit noticeably relative to TMVA's native DNN — test loss `0.099` vs train loss `0.245` at a given configuration — while TMVA's own CNN was the most stable across training. This difference is partly because 10 epochs on 1600 events is aggressive for a PyTorch model without regularisation, whereas TMVA's internal DNN applies its own early stopping logic.

**`TMVA_CNN_Classification`** — classification using a 2D convolutional network, with an optional path through Keras via the PyMVA interface. This was important context for the Keras parser work since PyMVA is the bridge that `RModelParser_Keras` depends on.

**`TMVA_SOFIE_Models.py`** — Keras to SOFIE pipeline. Trained a small Keras model, called `ParseFromMemory()` in SOFIE's Python interface, and generated `HiggsModel.hxx` and `KerasModel.hxx`. These files are in `tmva/sofie/exercise_outputs/`.

**`TMVA_SOFIE_ONNX.C`** — ran the ONNX parser tutorial end to end. Observed that SOFIE uses a graph reordering pass and a shared memory pool for intermediate tensors — two features that are specific to the ONNX path and not currently present in the PyTorch path.

**`TMVA_SOFIE_PyTorch.C`** — ran the PyTorch parser tutorial. The generated code for the same `Linear→ReLU→Linear→ReLU` network is structurally different from the ONNX-generated code: the PyTorch parser emits ReLU as a separate loop over elements, while the ONNX parser fuses Gemm and ReLU into a single block. This codegen inconsistency between the two parsers is exactly the kind of thing the GSoC project is meant to address. The tutorial output is saved at `tmva/sofie/exercise_outputs/exercise3_pytorch_output.txt`.

The generated files from all three SOFIE tutorials are in `tmva/sofie/exercise_outputs/`: `PyTorchModel.hxx`, `PyTorchModel.dat`, `KerasModel.hxx`, `HiggsModel.hxx`, and their associated `.dat` weight files.
