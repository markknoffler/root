# tests/test_recurrent.py
import torch.nn as nn
import numpy as np
import pytest
from sofie_pytorch_parser.core.parser import SOFIEPyTorchParser


@pytest.fixture
def parser():
    return SOFIEPyTorchParser()


class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(32, 64, num_layers=2)

class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(32, 64, num_layers=2, bidirectional=True)

class SimpleGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(32, 64, num_layers=1, bidirectional=False)


def test_rnn_weight_shapes(parser):
    result = parser.parse(SimpleRNN(), (1, 10, 32))
    # Layer 0: weight_ih (64, 32), Layer 1: weight_ih (64, 64) — hidden→hidden
    assert result["initializers"]["rnn_weight_ih_l0"].shape == (64, 32)
    assert result["initializers"]["rnn_weight_ih_l1"].shape == (64, 64)
    assert result["initializers"]["rnn_weight_hh_l0"].shape == (64, 64)

def test_lstm_gate_stacking(parser):
    result = parser.parse(SimpleLSTM(), (1, 10, 32))
    # 4 gates × hidden=64 = 256 rows; bidirectional → _reverse weights present
    assert result["initializers"]["lstm_weight_ih_l0"].shape == (256, 32)
    assert result["initializers"]["lstm_weight_ih_l0_reverse"].shape == (256, 32)
    ops = [o for o in result["operators"] if o["nodeType"] == "onnx::LSTM"]
    assert ops[0]["nodeAttributes"]["gate_order"] == "pytorch_ifgo"

def test_gru_gate_stacking_and_order_documented(parser):
    result = parser.parse(SimpleGRU(), (1, 10, 32))
    # 3 gates × hidden=64 = 192 rows
    assert result["initializers"]["gru_weight_ih_l0"].shape == (192, 32)
    ops = [o for o in result["operators"] if o["nodeType"] == "onnx::GRU"]
    # Gate order must be documented — critical for ONNX compatibility
    assert ops[0]["nodeAttributes"]["gate_order"] == "pytorch_rzn"

def test_all_weight_dtypes_float32(parser):
    result = parser.parse(SimpleLSTM(), (1, 10, 32))
    for name, arr in result["initializers"].items():
        assert arr.dtype == np.float32, f"{name} is {arr.dtype}, expected float32"

