import copy
import time
import numpy as np
import ROOT
from ROOT.TMVA.Experimental import SOFIE

from .config import extract_hls_config
from .schema import sofie_layer_dict
from .layers.relu import MakeHLSReLU
from .layers.elu import MakeHLSELU
from .layers.gemm import MakeHLSGemm
from .layers.reshape import MakeHLSReshape
from .layers.concat import MakeHLSConcat
from .layers.batchnorm import MakeHLSBatchNorm
from .layers.conv import MakeHLSConv
from .layers.pooling import MakeHLSPooling
from .layers.binary import MakeHLSBinary
from .layers.sigmoid import MakeHLSSigmoid
from .layers.tanh import MakeHLSTanh
from .layers.softmax import MakeHLSSoftmax
from .layers.swish import MakeHLSSwish
from .layers.leaky_relu import MakeHLSLeakyRelu
from .layers.selu import MakeHLSSeLU
from .layers.thresholdedrelu import MakeHLSThresholdedRelu


def MakeHLSActivation(layer):
    attributes = layer["layerAttributes"]
    fLayerActivation = str(attributes.get("Activation", attributes.get("activation", "")))
    if fLayerActivation == "ReLU":
        return MakeHLSReLU(layer)
    if fLayerActivation == "ELU":
        return MakeHLSELU(layer)
    if fLayerActivation == "SeLU":
        return MakeHLSSeLU(layer)
    if fLayerActivation == "Sigmoid":
        return MakeHLSSigmoid(layer)
    if fLayerActivation == "Tanh":
        return MakeHLSTanh(layer)
    if fLayerActivation == "Softmax":
        return MakeHLSSoftmax(layer)
    if fLayerActivation in ("Swish", "SiLU"):
        return MakeHLSSwish(layer)
    if fLayerActivation == "LeakyReLU":
        return MakeHLSLeakyRelu(layer)
    if fLayerActivation == "ThresholdedReLU":
        return MakeHLSThresholdedRelu(layer)
    else:
        raise Exception(
            "TMVA.SOFIE - HLS4ML activation " + fLayerActivation + " is not supported"
        )


mapHLS4MLLayer = {
    "ReLU": MakeHLSReLU,
    "relu": MakeHLSReLU,
    "ELU": MakeHLSELU,
    "elu": MakeHLSELU,
    "SeLU": MakeHLSSeLU,
    "selu": MakeHLSSeLU,
    "Sigmoid": MakeHLSSigmoid,
    "sigmoid": MakeHLSSigmoid,
    "Tanh": MakeHLSTanh,
    "tanh": MakeHLSTanh,
    "LeakyReLU": MakeHLSLeakyRelu,
    "leaky_relu": MakeHLSLeakyRelu,
    "ThresholdedReLU": MakeHLSThresholdedRelu,
    "Softmax": MakeHLSSoftmax,
    "softmax": MakeHLSSoftmax,
    "Swish": MakeHLSSwish,
    "swish": MakeHLSSwish,
    "Dense": MakeHLSGemm,
    "BatchNormalization": MakeHLSBatchNorm,
    "Conv2D": MakeHLSConv,
    "Conv1D": MakeHLSConv,
    "MaxPooling2D": MakeHLSPooling,
    "AveragePooling2D": MakeHLSPooling,
    "GlobalAveragePooling2D": MakeHLSPooling,
    "Reshape": MakeHLSReshape,
    "Flatten": MakeHLSReshape,
    "Concatenate": MakeHLSConcat,
    "Concat": MakeHLSConcat,
    "Add": MakeHLSBinary,
    "Subtract": MakeHLSBinary,
    "Multiply": MakeHLSBinary,
}


def _move_op(op):
    ROOT.SetOwnership(op, False)
    return ROOT.std.unique_ptr[type(op)](op)


def _add_blas_for_layer(rmodel, f_layer_type):
    if f_layer_type == "Dense":
        rmodel.AddBlasRoutines({"Gemm", "Gemv"})
    elif f_layer_type == "BatchNormalization":
        rmodel.AddBlasRoutines({"Copy", "Axpy"})
    elif f_layer_type in ("Conv1D", "Conv2D"):
        rmodel.AddBlasRoutines({"Gemm", "Axpy"})


def _register_initialised_tensors(rmodel, layer):
    # add initialised tensors from extracted cfg
    for tname, arr in (layer.get("initialisers") or {}).items():
        a = np.asarray(arr, dtype=np.float32)
        shape = [1] if a.ndim == 0 else list(a.shape)
        flat = np.ascontiguousarray(a.flatten(), dtype=np.float32)
        rmodel.AddInitializedTensor["float"](tname, shape, flat)


def add_layer_into_RModel(rmodel, layer_data):
    # add one layer into RModel
    layer_data = copy.deepcopy(layer_data)
    f_layer_type = layer_data["layerType"]
    attrs = layer_data["layerAttributes"]
    layer_name = attrs.get("name", "layer")

    if f_layer_type in ("Reshape", "Flatten"):
        target = attrs.get("target_shape")
        if target is None:
            target = [-1]
        target = np.ascontiguousarray(np.asarray(target, dtype="int64"))
        if f_layer_type == "Flatten":
            target = np.ascontiguousarray(np.asarray([-1], dtype="int64"))
        shape_name = layer_name + "_shape"
        rmodel.AddInitializedTensor["int64_t"](shape_name, [len(target)], target)

    if f_layer_type not in mapHLS4MLLayer and f_layer_type != "Activation":
        return rmodel

    inputs = list(layer_data["layerInput"])
    outputs = list(layer_data["layerOutput"])
    f_layer_output = outputs[0]
    channels_last = layer_data.get("channels_last", True)

    if f_layer_type == "GlobalAveragePooling2D":
        if channels_last:
            perm = [0, 3, 1, 2]
            if "_build_input_shape" in attrs:
                rank = len(attrs["_build_input_shape"])
                if rank == 3:
                    perm = [0, 2, 1]
            op = SOFIE.ROperator_Transpose("float")(perm, inputs[0], layer_name + "PreTrans")
            rmodel.AddOperator(_move_op(op))
            inputs[0] = layer_name + "PreTrans"
        outputs[0] = layer_name + "Squeeze"
        layer_data["layerInput"] = inputs
        layer_data["layerOutput"] = outputs
        rmodel.AddOperator(_move_op(mapHLS4MLLayer[f_layer_type](layer_data)))
        op = SOFIE.ROperator_Reshape(SOFIE.ReshapeOpMode.Squeeze, [2, 3], layer_name + "Squeeze", f_layer_output)
        rmodel.AddOperator(_move_op(op))
        return rmodel

    if f_layer_type == "BatchNormalization":
        build_shape = attrs.get("_build_input_shape")
        if not build_shape:
            raise RuntimeError("BatchNorm layer " + layer_name + " missing _build_input_shape in schema")
        rank = len(build_shape)
        axis = attrs.get("axis", -1)
        if isinstance(axis, (list, tuple)):
            axis = axis[0]
        axis = int(axis)
        if axis < 0:
            axis += rank
        f_attr_perm = list(range(0, rank))
        if axis < rank:
            f_attr_perm[1], f_attr_perm[axis] = f_attr_perm[axis], f_attr_perm[1]
        op = SOFIE.ROperator_Transpose("float")(f_attr_perm, inputs[0], layer_name + "PreTrans")
        rmodel.AddOperator(_move_op(op))
        inputs[0] = layer_name + "PreTrans"
        outputs[0] = layer_name + "PostTrans"
        layer_data["layerInput"] = inputs
        layer_data["layerOutput"] = outputs
        rmodel.AddOperator(_move_op(mapHLS4MLLayer[f_layer_type](layer_data)))
        op = SOFIE.ROperator_Transpose("float")(f_attr_perm, layer_name + "PostTrans", f_layer_output)
        rmodel.AddOperator(_move_op(op))
        return rmodel

    if f_layer_type in ("MaxPooling2D", "AveragePooling2D"):
        if channels_last:
            rank = len(attrs["_build_input_shape"]) if "_build_input_shape" in attrs else 4
            axis = attrs.get("axis", -1)
            if axis < 0:
                axis += rank
            perm = list(range(rank))
            if axis < rank:
                perm[1], perm[axis] = perm[axis], perm[1]
            op = SOFIE.ROperator_Transpose("float")(perm, inputs[0], layer_name + "PreTrans")
            rmodel.AddOperator(_move_op(op))
            inputs[0] = layer_name + "PreTrans"
            outputs[0] = layer_name + "PostTrans"
            layer_data["layerInput"] = inputs
            layer_data["layerOutput"] = outputs
            rmodel.AddOperator(_move_op(mapHLS4MLLayer[f_layer_type](layer_data)))
            op = SOFIE.ROperator_Transpose("float")(perm, layer_name + "PostTrans", f_layer_output)
            rmodel.AddOperator(_move_op(op))
        else:
            rmodel.AddOperator(_move_op(mapHLS4MLLayer[f_layer_type](layer_data)))
        return rmodel

    if f_layer_type == "Conv2D":
        if channels_last:
            perm = [0, 3, 1, 2]
            if "_build_input_shape" in attrs:
                rank = len(attrs["_build_input_shape"])
                if rank == 3:
                    perm = [0, 2, 1]
            op = SOFIE.ROperator_Transpose("float")(perm, inputs[0], layer_name + "PreTrans")
            rmodel.AddOperator(_move_op(op))
            inputs[0] = layer_name + "PreTrans"
            layer_data["layerInput"] = inputs
        outputs[0] = layer_name + "PostTrans"
        layer_data["layerOutput"] = outputs
        rmodel.AddOperator(_move_op(mapHLS4MLLayer[f_layer_type](layer_data)))
        if channels_last:
            perm = [0, 2, 3, 1]
            if "_build_input_shape" in attrs:
                rank = len(attrs["_build_input_shape"])
                if rank == 3:
                    perm = [0, 2, 1]
            op = SOFIE.ROperator_Transpose("float")(perm, layer_name + "PostTrans", f_layer_output)
            rmodel.AddOperator(_move_op(op))
        return rmodel

    if f_layer_type == "Conv1D":
        rmodel.AddOperator(_move_op(mapHLS4MLLayer[f_layer_type](layer_data)))
        return rmodel

    if f_layer_type == "Activation":
        op = MakeHLSActivation(layer_data)
        rmodel.AddOperator(_move_op(op))
        return rmodel

    op = mapHLS4MLLayer[f_layer_type](layer_data)
    rmodel.AddOperator(_move_op(op))
    return rmodel


def build_rmodel(cfg, name=None):
    # build RModel using cfg only
    model_name = name or cfg.get("name", "HLS4MLModel")
    parsetime = time.asctime(time.gmtime(time.time()))
    rmodel = SOFIE.RModel.RModel(model_name, parsetime)

    input_names = cfg.get("inputs", [])
    if not input_names:
        input_names = ["input_0"]
    input_shape = cfg.get("input_shape", [1])
    if len(input_shape) == 1:
        input_shape = [1, input_shape[0]]

    rmodel.AddInputTensorInfo(input_names[0], SOFIE.ETensorType.FLOAT, input_shape)
    rmodel.AddInputTensorName(input_names[0])
    rmodel.AddBlasRoutines({"Gemm", "Gemv"})

    for layer in cfg.get("layers", []):
        if layer.get("layerType") == "Input":
            continue
        lt = layer.get("layerType")
        _add_blas_for_layer(rmodel, lt)
        if lt in ("SeLU", "selu", "Sigmoid", "sigmoid"):
            rmodel.AddNeededStdLib("cmath")
        _register_initialised_tensors(rmodel, layer)
        rmodel = add_layer_into_RModel(rmodel, sofie_layer_dict(layer))

    output_names = list(cfg.get("outputs", []))
    if not output_names:
        layers = cfg.get("layers", [])
        for L in reversed(layers):
            outs = L.get("layerOutput", [])
            if outs:
                output_names = list(outs)
                break
    if output_names:
        rmodel.AddOutputTensorNameList(output_names)

    # Initialize model to register all intermediate tensors
    rmodel.Initialize()

    return rmodel


class PyHLS4ML:
    @staticmethod
    def ParseFromModelGraph(hls_model, name=None, keras_model=None):
        cfg = extract_hls_config(hls_model, keras_model=keras_model)
        return build_rmodel(cfg, name=name)

    @staticmethod
    def ExtractConfig(hls_model, keras_model=None):
        # extract canonical cfg
        return extract_hls_config(hls_model, keras_model=keras_model)

    @staticmethod
    def BuildFromConfig(cfg, name=None):
        # build from extracted cfg
        return build_rmodel(cfg, name=name)
