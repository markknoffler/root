import os
import time
import numpy as np
import ROOT
from ROOT.TMVA.Experimental import SOFIE

from .config import extract_hls_config
from .layers.relu import MakeHLSReLU
from .layers.elu import MakeHLSELU
from .layers.gemm import MakeHLSGemm
from .layers.reshape import MakeHLSReshape
from .layers.concat import MakeHLSConcat


def MakeHLSActivation(layer):
    attributes = layer["layerAttributes"]
    act = attributes.get("Activation", attributes.get("activation", "")).lower()
    if act in ("relu",):
        return MakeHLSReLU(layer)
    if act in ("elu",):
        return MakeHLSELU(layer)
    raise Exception("TMVA.SOFIE - HLS4ML activation " + str(act) + " is not supported")


mapHLS4MLLayer = {
    "ReLU": MakeHLSReLU,
    "relu": MakeHLSReLU,
    "ELU": MakeHLSELU,
    "elu": MakeHLSELU,
    "Dense": MakeHLSGemm,
    "Reshape": MakeHLSReshape,
    "Flatten": MakeHLSReshape,
    "Concatenate": MakeHLSConcat,
    "Concat": MakeHLSConcat,
}


def _layer_type_from_hls(hls_layer, cfg_layer):
    class_name = cfg_layer.get("class_name", type(hls_layer).__name__)
    attrs = cfg_layer.get("attributes", {})
    if "Dense" in class_name:
        return "Dense"
    if "Activation" in class_name:
        act = attrs.get("Activation", attrs.get("activation", ""))
        a = str(act).lower()
        if a in ("relu",):
            return "ReLU"
        if a in ("elu",):
            return "ELU"
        return "Activation"
    if "Reshape" in class_name or "Flatten" in class_name:
        return "Reshape" if "Reshape" in class_name else "Flatten"
    if "Concat" in class_name:
        return "Concatenate"
    if "Input" in class_name:
        return "Input"
    return None


def _build_layer_data(hls_layer, cfg_layer):
    layer_type = _layer_type_from_hls(hls_layer, cfg_layer)
    if layer_type is None:
        return None
    name = cfg_layer.get("name", getattr(hls_layer, "name", "layer"))
    inputs = cfg_layer.get("inputs", [])
    outputs = cfg_layer.get("outputs", [])
    if not inputs and hasattr(hls_layer, "inputs"):
        inputs = [getattr(x, "name", str(x)) for x in hls_layer.inputs]
    if not outputs and hasattr(hls_layer, "outputs"):
        outputs = [getattr(x, "name", str(x)) for x in hls_layer.outputs]
    layer_data = {
        "layerType": layer_type,
        "layerInput": list(inputs),
        "layerOutput": list(outputs),
        "layerDType": "float32",
        "layerAttributes": dict(cfg_layer.get("attributes", {})),
    }
    layer_data["layerAttributes"]["name"] = name
    if layer_type == "Dense":
        layer_data["layerWeight"] = [name + "_W", name + "_B"]
    if layer_type in ("Reshape", "Flatten"):
        shape = cfg_layer.get("output_shape")
        if shape is not None:
            try:
                layer_data["layerAttributes"]["target_shape"] = list(shape)
            except Exception:
                pass
        if not layer_data["layerAttributes"].get("target_shape"):
            if hasattr(hls_layer, "get_output_variable"):
                try:
                    var = hls_layer.get_output_variable()
                    if hasattr(var, "shape"):
                        layer_data["layerAttributes"]["target_shape"] = list(var.shape)
                except Exception:
                    pass
    if layer_type == "Concatenate":
        layer_data["layerAttributes"]["axis"] = layer_data["layerAttributes"].get("axis", 1)
    return layer_data


def _move_op(op):
    ROOT.SetOwnership(op, False)
    return ROOT.std.unique_ptr[type(op)](op)


def add_layer_into_RModel(rmodel, layer_data):
    fLayerType = layer_data["layerType"]
    if fLayerType in ("Reshape", "Flatten"):
        attrs = layer_data["layerAttributes"]
        name = attrs.get("name", "reshape")
        target = attrs.get("target_shape")
        if target is None:
            target = [-1]
        target = np.asarray(target, dtype="int64")
        if fLayerType == "Flatten":
            target = np.asarray([-1], dtype="int64")
        shape_name = name + "_shape"
        rmodel.AddInitializedTensor["int64_t"](shape_name, [len(target)], target.data)
    if fLayerType not in mapHLS4MLLayer and fLayerType != "Activation":
        return rmodel
    if fLayerType == "Activation":
        op = MakeHLSActivation(layer_data)
    else:
        op = mapHLS4MLLayer[fLayerType](layer_data)
    rmodel.AddOperator(_move_op(op))
    return rmodel


def _get_dense_weights_from_keras(keras_model, layer_name):
    w_arr = None
    b_arr = None
    for layer in keras_model.layers:
        if getattr(layer, "name", None) == layer_name:
            try:
                weights = layer.get_weights()
                if len(weights) >= 1:
                    w_arr = np.asarray(weights[0], dtype="float32")
                if len(weights) >= 2:
                    b_arr = np.asarray(weights[1], dtype="float32").flatten()
            except Exception:
                pass
            break
    return w_arr, b_arr


class PyHLS4ML:
    @staticmethod
    def ParseFromModelGraph(hls_model, name=None, keras_model=None):
        cfg = extract_hls_config(hls_model)
        model_name = name or cfg.get("name", "HLS4MLModel")
        parsetime = time.asctime(time.gmtime(time.time()))
        rmodel = SOFIE.RModel.RModel(model_name, parsetime)

        hls_layers = hls_model.get_layers() if hasattr(hls_model, "get_layers") else getattr(hls_model, "layers", [])
        cfg_layers = cfg.get("layers", [])

        input_names = cfg.get("inputs", [])
        if not input_names and hasattr(hls_model, "inputs"):
            input_names = [getattr(x, "name", str(x)) for x in hls_model.inputs]
        if not input_names:
            input_names = ["input_0"]

        input_shape = [1]
        if hasattr(hls_model, "inputs") and hls_model.inputs:
            inp = hls_model.inputs[0]
            if hasattr(inp, "shape"):
                try:
                    input_shape = list(inp.shape)
                except Exception:
                    pass
        if len(input_shape) == 1:
            input_shape = [1, input_shape[0]]
        rmodel.AddInputTensorInfo(input_names[0], SOFIE.ETensorType.FLOAT, input_shape)
        rmodel.AddInputTensorName(input_names[0])

        rmodel.AddBlasRoutines({"Gemm", "Gemv"})

        for i, (hls_layer, cfg_layer) in enumerate(zip(hls_layers, cfg_layers)):
            layer_data = _build_layer_data(hls_layer, cfg_layer)
            if layer_data is None or layer_data["layerType"] == "Input":
                continue
            if layer_data["layerType"] == "Dense":
                w_arr = None
                b_arr = None
                wdict = {}
                if hasattr(hls_layer, "get_weights"):
                    try:
                        w = hls_layer.get_weights()
                        if isinstance(w, dict):
                            wdict = w
                        elif isinstance(w, (list, tuple)) and len(w) >= 1:
                            wdict = {"weight": w[0], "bias": w[1] if len(w) > 1 else None}
                    except Exception:
                        pass
                if hasattr(hls_layer, "weights") and not wdict:
                    try:
                        wattr = hls_layer.weights
                        if hasattr(wattr, "items"):
                            wdict = dict(wattr)
                        elif hasattr(wattr, "values"):
                            vals = list(wattr.values())
                            if len(vals) >= 1:
                                wdict = {"weight": vals[0], "bias": vals[1] if len(vals) > 1 else None}
                    except Exception:
                        pass
                def _to_numpy(x):
                    if x is None:
                        return None
                    if hasattr(x, "value"):
                        x = x.value
                    if hasattr(x, "data"):
                        x = x.data
                    return np.asarray(x, dtype="float32")

                for k, v in wdict.items():
                    if v is None:
                        continue
                    ks = str(k).lower()
                    if "weight" in ks or "kernel" in ks or "w" == ks:
                        w_arr = _to_numpy(v)
                        if w_arr is not None:
                            w_arr = w_arr.copy()
                        break
                for k, v in wdict.items():
                    if v is None:
                        continue
                    ks = str(k).lower()
                    if "bias" in ks or "b" == ks:
                        b_arr = _to_numpy(v)
                        if b_arr is not None:
                            b_arr = b_arr.flatten()
                        break
                lname = layer_data["layerAttributes"]["name"]
                if w_arr is None and keras_model is not None:
                    w_arr, b_arr = _get_dense_weights_from_keras(keras_model, lname)
                if w_arr is None:
                    raise RuntimeError("Dense layer " + lname + " has no weights (pass keras_model= for Keras-sourced HLS4ML)")
                if b_arr is None:
                    b_arr = np.zeros(w_arr.shape[0], dtype="float32")
                rmodel.AddInitializedTensor["float"](lname + "_W", list(w_arr.shape), w_arr.data)
                rmodel.AddInitializedTensor["float"](lname + "_B", [len(b_arr)], b_arr.data)
            rmodel = add_layer_into_RModel(rmodel, layer_data)

        output_names = cfg.get("outputs", [])
        if not output_names and cfg_layers:
            out_layer = cfg_layers[-1]
            output_names = out_layer.get("outputs", [])
        if output_names:
            rmodel.AddOutputTensorNameList(output_names)
        else:
            last_out = None
            for ld in reversed(cfg_layers):
                outs = ld.get("outputs", [])
                if outs:
                    last_out = outs[0]
                    break
            if last_out:
                rmodel.AddOutputTensorNameList([last_out])

        return rmodel
