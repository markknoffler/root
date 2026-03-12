# core/rmodel_builder.py
# Builds SOFIE RModel directly from parsed Python dict (no JSON).
# Mirrors the Keras parser: dictionary → RModel via PyROOT.
# Backwards compatibility: JSON → C++ ParseFromPython is unchanged.

import time
import copy
from typing import Dict, List, Any

import numpy as np


def _move_op(op):
    import ROOT
    ROOT.SetOwnership(op, False)
    return ROOT.std.unique_ptr[type(op)](op)


def _make_gemm(node: Dict) -> Any:
    from ROOT.TMVA.Experimental import SOFIE
    attrs = node["nodeAttributes"]
    inp = node["nodeInputs"]
    out = node["nodeOutputs"][0]
    dtype = node["nodeDType"][0]
    alpha = float(attrs.get("alpha", 1.0))
    beta = float(attrs.get("beta", 1.0))
    transA = int(attrs.get("transA", 0))
    transB = int(attrs.get("transB", 1))
    if SOFIE.ConvertStringToType(dtype) == SOFIE.ETensorType.FLOAT:
        return SOFIE.ROperator_Gemm["float"](alpha, beta, transA, transB, inp[0], inp[1], inp[2], out)
    raise RuntimeError("Gemm does not support dtype " + dtype)


def _make_relu(node: Dict) -> Any:
    from ROOT.TMVA.Experimental import SOFIE
    inp = node["nodeInputs"][0]
    out = node["nodeOutputs"][0]
    return SOFIE.ROperator_Relu["float"](inp, out)


def _make_sigmoid(node: Dict) -> Any:
    from ROOT.TMVA.Experimental import SOFIE
    inp = node["nodeInputs"][0]
    out = node["nodeOutputs"][0]
    return SOFIE.ROperator_Sigmoid["float"](inp, out)


def _make_elu(node: Dict) -> Any:
    from ROOT.TMVA.Experimental import SOFIE
    attrs = node["nodeAttributes"]
    inp = node["nodeInputs"][0]
    out = node["nodeOutputs"][0]
    alpha = float(attrs.get("alpha", 1.0))
    return SOFIE.ROperator_Elu["float"](alpha, inp, out)


def _make_conv(node: Dict) -> Any:
    from ROOT.TMVA.Experimental import SOFIE
    attrs = node["nodeAttributes"]
    inp = node["nodeInputs"]
    out = node["nodeOutputs"][0]
    nameX, nameW, nameB = inp[0], inp[1], inp[2]
    autopad = "NOTSET"
    dilations = list(attrs.get("dilations", [1, 1]))
    group = int(attrs.get("group", 1))
    kernel = list(attrs.get("kernel_shape", [3, 3]))
    pads = list(attrs.get("pads", [0, 0, 0, 0]))
    strides = list(attrs.get("strides", [1, 1]))
    return SOFIE.ROperator_Conv["float"](
        autopad, dilations, group, kernel, pads, strides,
        nameX, nameW, nameB, out
    )


def _make_maxpool(node: Dict) -> Any:
    from ROOT.TMVA.Experimental import SOFIE
    attrs = node["nodeAttributes"]
    inp = node["nodeInputs"][0]
    out = node["nodeOutputs"][0]
    attr = SOFIE.RAttributes_Pool()
    attr.auto_pad = "NOTSET"
    attr.ceil_mode = int(attrs.get("ceil_mode", 0))
    attr.dilations = list(attrs.get("dilations", [1, 1]))
    attr.kernel_shape = list(attrs.get("kernel_shape", [2, 2]))
    attr.pads = list(attrs.get("pads", [0, 0, 0, 0]))
    attr.strides = list(attrs.get("strides", [1, 1]))
    return SOFIE.ROperator_Pool["float"](SOFIE.PoolOpMode.MaxPool, attr, inp, out)


def _make_batchnorm(node: Dict) -> Any:
    from ROOT.TMVA.Experimental import SOFIE
    attrs = node["nodeAttributes"]
    inp = node["nodeInputs"]
    out = node["nodeOutputs"][0]
    epsilon = float(attrs.get("epsilon", 1e-5))
    momentum = float(attrs.get("momentum", 0.1))
    training = int(attrs.get("training_mode", 0))
    return SOFIE.ROperator_BatchNormalization["float"](
        epsilon, momentum, training,
        inp[0], inp[1], inp[2], inp[3], inp[4], out
    )


def _make_rnn(node: Dict) -> Any:
    from ROOT.TMVA.Experimental import SOFIE
    attrs = node["nodeAttributes"]
    inp = node["nodeInputs"]
    out = node["nodeOutputs"][0]
    hidden = int(attrs.get("hidden_size", 16))
    bidi = int(attrs.get("bidirectional", 0))
    direction = "bidirectional" if bidi else "forward"
    nonlin = str(attrs.get("nonlinearity", "tanh")).lower()
    activation = "Tanh" if nonlin == "tanh" else "Relu"
    nameB = inp[3] if len(inp) > 3 else ""
    return SOFIE.ROperator_RNN["float"](
        [], [], [activation], 0.0, direction, hidden, 0,
        inp[0], inp[1], inp[2], nameB, "", "", out, ""
    )


def _make_lstm(node: Dict) -> Any:
    from ROOT.TMVA.Experimental import SOFIE
    attrs = node["nodeAttributes"]
    inp = node["nodeInputs"]
    out = node["nodeOutputs"][0]
    hidden = int(attrs.get("hidden_size", 16))
    bidi = int(attrs.get("bidirectional", 0))
    direction = "bidirectional" if bidi else "forward"
    nameB = inp[3] if len(inp) > 3 else ""
    return SOFIE.ROperator_LSTM["float"](
        [], [], ["Sigmoid", "Tanh", "Tanh"], 0.0, direction, hidden, 0, 0,
        inp[0], inp[1], inp[2], nameB, "", "", "", "", out, "", ""
    )


def _make_gru(node: Dict) -> Any:
    from ROOT.TMVA.Experimental import SOFIE
    attrs = node["nodeAttributes"]
    inp = node["nodeInputs"]
    out = node["nodeOutputs"][0]
    hidden = int(attrs.get("hidden_size", 16))
    bidi = int(attrs.get("bidirectional", 0))
    direction = "bidirectional" if bidi else "forward"
    nameB = inp[3] if len(inp) > 3 else ""
    return SOFIE.ROperator_GRU["float"](
        [], [], ["Sigmoid", "Tanh"], 0.0, direction, hidden, 0, 0,
        inp[0], inp[1], inp[2], nameB, "", "", out, ""
    )


_MAP_NODE_TO_MAKER = {
    "onnx::Gemm": _make_gemm,
    "onnx::Relu": _make_relu,
    "onnx::Sigmoid": _make_sigmoid,
    "onnx::Elu": _make_elu,
    "onnx::Conv": _make_conv,
    "onnx::MaxPool": _make_maxpool,
    "onnx::BatchNormalization": _make_batchnorm,
    "onnx::RNN": _make_rnn,
    "onnx::LSTM": _make_lstm,
    "onnx::GRU": _make_gru,
}


def build_rmodel(
    parsed: Dict,
    model_name: str = "PyTorchModel",
) -> Any:
    """
    Build SOFIE RModel from parsed Python dict (no JSON, no C++).
    Uses PyROOT to create RModel and add operators/initializers directly.
    Mirrors the Keras parser's direct Python → RModel flow.

    Args:
        parsed: Dict with keys operators, initializers, inputs, outputs
        model_name: Name for the RModel

    Returns:
        SOFIE.RModel ready for Generate() and OutputGenerated()
    """
    from ROOT.TMVA.Experimental import SOFIE

    operators = copy.deepcopy(parsed["operators"])
    initializers = dict(parsed["initializers"])
    inputs = parsed["inputs"]
    outputs = parsed["outputs"]

    parsetime = time.asctime(time.gmtime(time.time()))
    rmodel = SOFIE.RModel.RModel(model_name, parsetime)

    input_name = list(inputs.keys())[0]
    input_shape = list(inputs[input_name])
    rmodel.AddInputTensorInfo(input_name, SOFIE.ETensorType.FLOAT, input_shape)
    rmodel.AddInputTensorName(input_name)

    rmodel.AddBlasRoutines({"Gemm", "Gemv"})

    for node in operators:
        if node["nodeType"] == "onnx::Conv" and len(node["nodeInputs"]) == 2:
            nameW = node["nodeInputs"][1]
            w = initializers.get(nameW)
            if w is not None:
                nameB = node["nodeOutputs"][0] + "_zero_bias"
                initializers[nameB] = np.zeros(w.shape[0], dtype=np.float32)
                node["nodeInputs"].append(nameB)

    for name, arr in initializers.items():
        shape = list(arr.shape) if hasattr(arr, "shape") else [len(arr)]
        arr_flat = np.ascontiguousarray(np.asarray(arr, dtype=np.float32).flatten())
        rmodel.AddInitializedTensor["float"](name, shape, arr_flat)

    for node in operators:
        ntype = node["nodeType"]
        if ntype == "onnx::Gemm":
            rmodel.AddBlasRoutines({"Gemm", "Gemv"})
        elif ntype == "onnx::Conv":
            rmodel.AddBlasRoutines({"Gemm", "Axpy"})
        elif ntype in ("onnx::RNN", "onnx::LSTM", "onnx::GRU"):
            rmodel.AddBlasRoutines({"Gemm", "Axpy"})
        maker = _MAP_NODE_TO_MAKER.get(ntype)
        if maker is None:
            raise RuntimeError("TMVA::SOFIE - PyTorch node " + ntype + " not yet supported for Python RModel build")
        op = maker(node)
        rmodel.AddOperator(_move_op(op))

    output_name = list(outputs.keys())[0]
    rmodel.AddOutputTensorNameList([output_name])

    return rmodel
