"""
Test parse_to_rmodel — Dictionary-based Python → RModel (no JSON, no C++).
Demonstrates direct PyTorch → SOFIE flow, mirroring Keras parser behavior.
Run: python tmva/sofie_pytorch_parser/tests/test_parse_to_rmodel.py
Requires: ROOT with SOFIE, PyTorch

IMPORTANT: ROOT must be imported before PyTorch to avoid segfaults from
library loading order conflicts.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

# Load ROOT/SOFIE first (before PyTorch) to avoid segfaults
try:
    import ROOT
    ROOT.gSystem.Load("libROOTTMVASofie")
    from ROOT.TMVA.Experimental import SOFIE
except Exception as e:
    print("SKIP: ROOT/SOFIE not available:", e)
    sys.exit(0)

import torch
import torch.nn as nn
from tmva.sofie_pytorch_parser import parse_to_rmodel

_repo = os.path.normpath(os.path.join(os.path.dirname(__file__), "../..", ".."))
OUT = os.path.join(_repo, "tmva", "sofie", "exercise_outputs")
os.makedirs(OUT, exist_ok=True)


def test_tutorial_model():
    """Tutorial dense model: Linear→ReLU→Linear→ReLU — parse_to_rmodel and generate .hxx"""
    print("\n=== parse_to_rmodel: Tutorial Dense Model ===")
    model = nn.Sequential(
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU()
    )
    model.eval()

    rmodel = parse_to_rmodel(model, input_shape=(2, 32), model_name="TutorialModel_PythonRModel")
    rmodel.Generate()
    out_path = os.path.join(OUT, "TutorialModel_PythonRModel.hxx")
    rmodel.OutputGenerated(out_path)

    print(f"  Generated: {out_path}")
    assert os.path.exists(out_path)
    assert os.path.exists(out_path.replace(".hxx", ".dat"))
    print("  PASS: parse_to_rmodel → RModel → .hxx + .dat")


def test_elu_model():
    """ELU model — parse_to_rmodel"""
    print("\n=== parse_to_rmodel: ELU Model ===")
    model = nn.Sequential(nn.Linear(16, 8), nn.ELU(alpha=1.5))
    model.eval()

    rmodel = parse_to_rmodel(model, input_shape=(1, 16), model_name="ELUModel_PythonRModel")
    rmodel.Generate()
    out_path = os.path.join(OUT, "ELUModel_PythonRModel.hxx")
    rmodel.OutputGenerated(out_path)

    print(f"  Generated: {out_path}")
    assert os.path.exists(out_path)
    print("  PASS: ELU model via parse_to_rmodel")


def main():
    print("Testing parse_to_rmodel (Python dict → RModel via PyROOT)")
    test_tutorial_model()
    test_elu_model()
    print("\n" + "=" * 50)
    print("parse_to_rmodel tests PASSED")


if __name__ == "__main__":
    main()
