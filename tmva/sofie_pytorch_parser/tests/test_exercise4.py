"""
Exercise 4 — Test all 6 new operator parsers
Run: python tmva/sofie_pytorch_parser/tests/test_exercise4.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import torch
import torch.nn as nn
import numpy as np
from tmva.sofie_pytorch_parser.core.parser import SOFIEPyTorchParser
from tmva.sofie_pytorch_parser.core.exporter import export_json

parser = SOFIEPyTorchParser()
OUT = "tmva/sofie/exercise3_outputs"
os.makedirs(OUT, exist_ok=True)

def check(parsed, op_name, label):
    ops = [op["nodeType"] for op in parsed["operators"]]
    found = any(op_name.lower() in o.lower() for o in ops)
    status = "PASS" if found else "FAIL"
    print(f"  [{status}] {label}: found ops = {ops}")
    assert found, f"{op_name} not found in {ops}"

# ── 1. ELU ──────────────────────────────────────────────────────────────────
print("\n=== 1. ELU ===")
model = nn.Sequential(nn.Linear(16, 8), nn.ELU(alpha=1.5))
model.eval()
parsed = parser.parse(model, input_shape=(1, 16))
check(parsed, "Elu", "ELU alpha=1.5")
export_json(parsed, f"{OUT}/elu_model.json")
print(f"  Operators: {[op['nodeType'] for op in parsed['operators']]}")
print(f"  Weights:   {list(parsed['initializers'].keys())}")

# ── 2. MaxPool2D ─────────────────────────────────────────────────────────────
print("\n=== 2. MaxPool2D ===")
model = nn.Sequential(
    nn.Conv2d(1, 4, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2)
)
model.eval()
parsed = parser.parse(model, input_shape=(1, 1, 8, 8))
check(parsed, "MaxPool", "MaxPool2D k=2 s=2")
export_json(parsed, f"{OUT}/maxpool_model.json")
print(f"  Operators: {[op['nodeType'] for op in parsed['operators']]}")

# ── 3. BatchNorm2D ───────────────────────────────────────────────────────────
print("\n=== 3. BatchNorm2D ===")
model = nn.Sequential(
    nn.Conv2d(1, 4, kernel_size=3, padding=1),
    nn.BatchNorm2d(4),
    nn.ReLU()
)
model.eval()
parsed = parser.parse(model, input_shape=(1, 1, 8, 8))
check(parsed, "BatchNorm", "BatchNorm2D")
export_json(parsed, f"{OUT}/batchnorm_model.json")
print(f"  Operators: {[op['nodeType'] for op in parsed['operators']]}")
print(f"  BN weights: {[k for k in parsed['initializers'] if 'bn' in k.lower() or 'batch' in k.lower() or 'norm' in k.lower()]}")

# ── 4. RNN ───────────────────────────────────────────────────────────────────
print("\n=== 4. RNN ===")
for nonlin in ["tanh", "relu"]:
    model = nn.RNN(input_size=8, hidden_size=16, nonlinearity=nonlin, batch_first=True)
    model.eval()
    parsed = parser.parse(model, input_shape=(1, 5, 8))
    check(parsed, "RNN", f"RNN nonlinearity={nonlin}")
    export_json(parsed, f"{OUT}/rnn_{nonlin}_model.json")

# Bidirectional RNN
model = nn.RNN(input_size=8, hidden_size=16, bidirectional=True, batch_first=True)
model.eval()
parsed = parser.parse(model, input_shape=(1, 5, 8))
check(parsed, "RNN", "RNN bidirectional")
attrs = parsed["operators"][0]["nodeAttributes"]
print(f"  direction={attrs.get('direction')}, hiddensize={attrs.get('hiddensize')}")

# ── 5. LSTM ──────────────────────────────────────────────────────────────────
print("\n=== 5. LSTM ===")
model = nn.LSTM(input_size=8, hidden_size=16, num_layers=1, batch_first=True)
model.eval()
parsed = parser.parse(model, input_shape=(1, 5, 8))
check(parsed, "LSTM", "LSTM standard")
export_json(parsed, f"{OUT}/lstm_model.json")
attrs = parsed["operators"][0]["nodeAttributes"]
print(f"  hiddensize={attrs.get('hiddensize')}, direction={attrs.get('direction')}")
print(f"  weights: {list(parsed['initializers'].keys())}")

# Bidirectional LSTM
model = nn.LSTM(input_size=8, hidden_size=16, bidirectional=True, batch_first=True)
model.eval()
parsed = parser.parse(model, input_shape=(1, 5, 8))
check(parsed, "LSTM", "LSTM bidirectional")
export_json(parsed, f"{OUT}/lstm_bidi_model.json")

# ── 6. GRU ───────────────────────────────────────────────────────────────────
print("\n=== 6. GRU ===")
model = nn.GRU(input_size=8, hidden_size=16, batch_first=True)
model.eval()
parsed = parser.parse(model, input_shape=(1, 5, 8))
check(parsed, "GRU", "GRU standard")
export_json(parsed, f"{OUT}/gru_model.json")
attrs = parsed["operators"][0]["nodeAttributes"]
print(f"  hiddensize={attrs.get('hiddensize')}, linear_before_reset={attrs.get('linearbeforereset')}")

model = nn.GRU(input_size=8, hidden_size=16, bidirectional=True, batch_first=True)
model.eval()
parsed = parser.parse(model, input_shape=(1, 5, 8))
check(parsed, "GRU", "GRU bidirectional")
export_json(parsed, f"{OUT}/gru_bidi_model.json")

print("\n" + "="*50)
print("ALL 6 OPERATORS PASSED")
print("JSON files written to", OUT)

