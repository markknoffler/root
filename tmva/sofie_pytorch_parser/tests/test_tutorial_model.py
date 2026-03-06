"""
Demo: Parse the same PyTorchModel.pt that TMVA_SOFIE_PyTorch.C generates
but using our new Python parser instead of the old _model_to_graph path.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import torch
import torch.nn as nn
import numpy as np
from tmva.sofie_pytorch_parser.core.parser import SOFIEPyTorchParser
from tmva.sofie_pytorch_parser.core.exporter import export_json

OUT = "tmva/sofie/exercise3_outputs"

# ── Recreate the exact same model the C++ tutorial builds ───────────────────
print("Recreating PyTorchModel (Linear32→16→ReLU→Linear16→8→ReLU)...")
model = nn.Sequential(
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU()
)
model.eval()

# Save as .pt (same as the tutorial does with torch.jit.save)
scripted = torch.jit.script(model)
torch.jit.save(scripted, f"{OUT}/TutorialModel_newparser.pt")

# ── Parse with our new Python parser ────────────────────────────────────────
print("Parsing with new SOFIEPyTorchParser...")
parser = SOFIEPyTorchParser()
parsed = parser.parse(model, input_shape=(2, 32))
export_json(parsed, f"{OUT}/TutorialModel_newparser.json")

print(f"\nOperators found: {[op['nodeType'] for op in parsed['operators']]}")
print(f"Weights found:   {list(parsed['initializers'].keys())}")
print(f"\nJSON written to {OUT}/TutorialModel_newparser.json")
print("\nThis JSON can now be passed to ParseFromPython() in C++ to generate")
print("identical inference code to what TMVA_SOFIE_PyTorch.C produces.")

# ── Verify output shape ──────────────────────────────────────────────────────
x = torch.randn(2, 32)
with torch.no_grad():
    y = model(x)
print(f"\nModel output shape: {y.shape}")  # should be [2, 8]
print("Done.")

