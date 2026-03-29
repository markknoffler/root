# TMVA hls4ml support assets

The **hls4ml → SOFIE** Python parser lives in the main ROOT PyROOT tree (same level as the Keras SOFIE parser):

`bindings/pyroot/pythonizations/python/ROOT/_pythonization/_tmva/_sofie/_parser/_hls4ml/`

After building ROOT with TMVA, use it from Python as:

```python
import ROOT
ROOT.gSystem.Load("libROOTTMVASofie")
rmodel = ROOT.TMVA.Experimental.SOFIE.PyHLS4ML.ParseFromModelGraph(hls_model, name="MyModel", keras_model=keras_model)
```

Optional split API: `PyHLS4ML.ExtractConfig(...)` then `PyHLS4ML.BuildFromConfig(cfg, name=...)`.

**Unit tests:** `bindings/pyroot/pythonizations/test/sofie_hls4ml_parser.py` (requires optional Python packages; see CMake `ROOT_ADD_PYUNITTEST`).

This directory may contain shared assets (for example ONNX fixtures) used by local experiments; it is not the home of the parser package.
