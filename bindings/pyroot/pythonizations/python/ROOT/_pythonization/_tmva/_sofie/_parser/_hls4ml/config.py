from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _get_keras_layer(keras_model: Any, layer_name: str) -> Any:
    if keras_model is None:
        return None
    for layer in keras_model.layers:
        if getattr(layer, "name", None) == layer_name:
            return layer
    return None


def _coerce_tuple(val: Any, default: Tuple[int, ...]) -> Tuple[int, ...]:
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        return tuple(int(x) for x in val)
    if isinstance(val, str):
        try:
            p = ast.literal_eval(val.strip())
            if isinstance(p, (list, tuple)):
                return tuple(int(x) for x in p)
        except Exception:
            pass
    try:
        return (int(val),)
    except Exception:
        return default


def _activation_type_from_string(act: Any) -> Optional[str]:
    a = str(act).lower().strip()
    if a in ("relu",):
        return "ReLU"
    if a in ("elu",):
        return "ELU"
    if a in ("selu",):
        return "SeLU"
    if a in ("sigmoid",):
        return "Sigmoid"
    if a in ("tanh",):
        return "Tanh"
    if a in ("softmax",):
        return "Softmax"
    if a in ("swish", "silu"):
        return "Swish"
    if a in ("leaky_relu", "leakyrelu"):
        return "LeakyReLU"
    if a in ("linear", "none"):
        return None
    return None


def _layer_type_from_hls(hls_layer: Any, cfg_layer: Dict[str, Any]) -> Optional[str]:
    class_name = cfg_layer.get("class_name", type(hls_layer).__name__)
    attrs = cfg_layer.get("attributes", {})

    if "Dense" in class_name and "Conv" not in class_name:
        return "Dense"
    if "BatchNormalization" in class_name or "BatchNorm" in class_name:
        return "BatchNormalization"
    if "Conv2D" in class_name:
        return "Conv2D"
    if "Conv1D" in class_name:
        return "Conv1D"
    if "MaxPooling2D" in class_name:
        return "MaxPooling2D"
    if "AveragePooling2D" in class_name:
        return "AveragePooling2D"
    if "GlobalAveragePooling2D" in class_name:
        return "GlobalAveragePooling2D"
    if class_name in ("Add", "Subtract", "Multiply"):
        return class_name
    if "Concat" in class_name:
        return "Concatenate"
    if "Reshape" in class_name:
        return "Reshape"
    if "Flatten" in class_name:
        return "Flatten"
    if class_name == "InputLayer":
        return "Input"
    if class_name == "ReLU":
        return "ReLU"
    if class_name == "ELU":
        return "ELU"
    if class_name == "LeakyReLU":
        return "LeakyReLU"
    if "Activation" in class_name:
        act = attrs.get("Activation", attrs.get("activation", ""))
        return _activation_type_from_string(act) or "Activation"

    return None


def _get_dense_weights_from_keras(keras_model: Any, layer_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    w_arr = None
    b_arr = None
    layer = _get_keras_layer(keras_model, layer_name)
    if layer is None:
        return w_arr, b_arr
    try:
        weights = layer.get_weights()
        if len(weights) >= 1:
            w_arr = np.asarray(weights[0], dtype=np.float32)
        if len(weights) >= 2:
            b_arr = np.asarray(weights[1], dtype=np.float32).flatten()
    except Exception:
        pass
    return w_arr, b_arr


def _get_bn_weights_from_keras(keras_model: Any, layer_name: str) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
]:
    layer = _get_keras_layer(keras_model, layer_name)
    if layer is None:
        return None, None, None, None
    try:
        w = layer.get_weights()
        if len(w) >= 4:
            return (
                np.asarray(w[0], dtype=np.float32),
                np.asarray(w[1], dtype=np.float32).flatten(),
                np.asarray(w[2], dtype=np.float32).flatten(),
                np.asarray(w[3], dtype=np.float32).flatten(),
            )
    except Exception:
        pass
    return None, None, None, None


def _get_conv_weights_from_keras(keras_model: Any, layer_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    layer = _get_keras_layer(keras_model, layer_name)
    if layer is None:
        return None, None
    try:
        w = layer.get_weights()
        if len(w) < 1:
            return None, None
        kernel = np.asarray(w[0], dtype=np.float32)
        bias = np.asarray(w[1], dtype=np.float32).flatten() if len(w) >= 2 else None
        if kernel.ndim == 4:
            kernel = np.transpose(kernel, (3, 2, 0, 1)).copy()
        elif kernel.ndim == 3:
            kernel = np.transpose(kernel, (2, 1, 0)).copy()
        if bias is None:
            bias = np.zeros(kernel.shape[0], dtype=np.float32)
        return kernel, bias
    except Exception:
        return None, None


def _to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if hasattr(x, "value"):
        x = x.value
    if hasattr(x, "data"):
        x = x.data
    return np.asarray(x, dtype=np.float32)


def _dense_weights_from_hls(hls_layer: Any) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
    wdict: Dict[str, Any] = {}
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
    w_arr = None
    b_arr = None
    for k, v in wdict.items():
        if v is None:
            continue
        ks = str(k).lower()
        if "weight" in ks or "kernel" in ks or ks == "w":
            w_arr = _to_numpy(v)
            if w_arr is not None:
                w_arr = w_arr.copy()
            break
    for k, v in wdict.items():
        if v is None:
            continue
        ks = str(k).lower()
        if "bias" in ks or ks == "b":
            b_arr = _to_numpy(v)
            if b_arr is not None:
                b_arr = b_arr.flatten()
            break
    return wdict, w_arr, b_arr


def _parse_precision_token(s: Any) -> Optional[Dict[str, int]]:
    if s is None:
        return None
    m = re.search(r"ap_fixed\s*<\s*(\d+)\s*,\s*(\d+)\s*>", str(s), re.I)
    if not m:
        m = re.search(r"fixed\s*<\s*(\d+)\s*,\s*(\d+)\s*>", str(s), re.I)
    if m:
        return {"bits": int(m.group(1)), "int_bits": int(m.group(2))}
    return None


def _precision_from_layer(hls_layer: Any, attrs: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ("result_t", "weight_t", "bias_t", "accum_t"):
        v = attrs.get(key)
        parsed = _parse_precision_token(v)
        if parsed:
            out[key] = parsed
    if not out and hasattr(hls_layer, "get_output_variable"):
        try:
            var = hls_layer.get_output_variable()
            if hasattr(var, "type"):
                parsed = _parse_precision_token(getattr(var, "type", None))
                if parsed:
                    out["output"] = parsed
        except Exception:
            pass
    return out


def _infer_input_shape(hls_model: Any) -> List[int]:
    input_shape: List[int] = [1]
    if hasattr(hls_model, "inputs") and hls_model.inputs:
        inp = hls_model.inputs[0]
        if hasattr(inp, "shape"):
            try:
                input_shape = list(inp.shape)
            except Exception:
                pass
    if len(input_shape) == 1:
        input_shape = [1, input_shape[0]]
    return input_shape


def _fill_conv_attributes_from_keras(canonical: Dict[str, Any], kl: Any) -> None:
    attrs = canonical["layerAttributes"]
    if hasattr(kl, "kernel_size"):
        ks = kl.kernel_size
        attrs["kernel_size"] = list(_coerce_tuple(ks, (1, 1))) if not isinstance(ks, int) else [int(ks)]
    if hasattr(kl, "strides"):
        st = kl.strides
        attrs["strides"] = list(_coerce_tuple(st, (1, 1))) if not isinstance(st, int) else [int(st)]
    if hasattr(kl, "padding"):
        attrs["padding"] = str(kl.padding)
    if hasattr(kl, "dilation_rate"):
        dr = kl.dilation_rate
        attrs["dilation_rate"] = list(dr) if isinstance(dr, (list, tuple)) else [int(dr)]
    if hasattr(kl, "groups"):
        attrs["groups"] = int(kl.groups)
    if hasattr(kl, "data_format"):
        canonical["channels_last"] = str(kl.data_format) == "channels_last"


def _fill_pool_attributes_from_keras(canonical: Dict[str, Any], kl: Any) -> None:
    attrs = canonical["layerAttributes"]
    if hasattr(kl, "pool_size"):
        attrs["pool_size"] = list(_coerce_tuple(kl.pool_size, (2, 2)))
    if hasattr(kl, "strides"):
        attrs["strides"] = list(_coerce_tuple(kl.strides, (2, 2)))
    if hasattr(kl, "padding"):
        attrs["padding"] = str(kl.padding)
    if hasattr(kl, "data_format"):
        canonical["channels_last"] = str(kl.data_format) == "channels_last"


def _canonicalize_layer(
    hls_layer: Any,
    cfg_layer: Dict[str, Any],
    keras_model: Any,
    tensor_shapes: Dict[str, List[int]],
) -> Optional[Dict[str, Any]]:
    layer_type = _layer_type_from_hls(hls_layer, cfg_layer)
    if layer_type is None:
        return None
    name = cfg_layer.get("name", getattr(hls_layer, "name", "layer"))
    inputs = list(cfg_layer.get("inputs", []))
    outputs = list(cfg_layer.get("outputs", []))
    if not inputs and hasattr(hls_layer, "inputs"):
        inputs = [getattr(x, "name", str(x)) for x in hls_layer.inputs]
    if not outputs and hasattr(hls_layer, "outputs"):
        outputs = [getattr(x, "name", str(x)) for x in hls_layer.outputs]

    attrs = dict(cfg_layer.get("attributes", {}))
    attrs["name"] = name

    canonical: Dict[str, Any] = {
        "layerType": layer_type,
        "layerInput": inputs,
        "layerOutput": outputs,
        "layerDType": "float32",
        "layerAttributes": attrs,
        "initialisers": {},
        "precision": _precision_from_layer(hls_layer, attrs),
        "name": name,
        "class_name": cfg_layer.get("class_name", type(hls_layer).__name__),
        "channels_last": True,
    }

    in_name = inputs[0] if inputs else None
    if in_name and in_name in tensor_shapes:
        canonical["layerAttributes"]["_build_input_shape"] = list(tensor_shapes[in_name])

    if layer_type == "Dense":
        canonical["layerWeight"] = [name + "_W", name + "_B"]
        _, w_arr, b_arr = _dense_weights_from_hls(hls_layer)
        if w_arr is None and keras_model is not None:
            w_arr, b_arr = _get_dense_weights_from_keras(keras_model, name)
        if w_arr is None:
            raise RuntimeError(
                "Dense layer " + name + " has no weights (pass keras_model= for Keras-sourced HLS4ML)"
            )
        if b_arr is None:
            b_arr = np.zeros(w_arr.shape[0], dtype=np.float32)
        canonical["initialisers"][name + "_W"] = np.ascontiguousarray(w_arr, dtype=np.float32)
        canonical["initialisers"][name + "_B"] = np.ascontiguousarray(b_arr.flatten(), dtype=np.float32)

    elif layer_type == "BatchNormalization":
        canonical["layerWeight"] = [name + "_scale", name + "_bias", name + "_mean", name + "_var"]
        g, b, m, v = _get_bn_weights_from_keras(keras_model, name)
        if g is None:
            raise RuntimeError("BatchNorm " + name + " needs keras_model= to load gamma/beta/mean/variance")
        canonical["initialisers"][name + "_scale"] = np.ascontiguousarray(g.flatten(), dtype=np.float32)
        canonical["initialisers"][name + "_bias"] = np.ascontiguousarray(b.flatten(), dtype=np.float32)
        canonical["initialisers"][name + "_mean"] = np.ascontiguousarray(m.flatten(), dtype=np.float32)
        canonical["initialisers"][name + "_var"] = np.ascontiguousarray(v.flatten(), dtype=np.float32)
        kl = _get_keras_layer(keras_model, name)
        if kl is not None:
            canonical["layerAttributes"]["epsilon"] = float(getattr(kl, "epsilon", 1e-3))
            canonical["layerAttributes"]["momentum"] = float(getattr(kl, "momentum", 0.99))
            ax = getattr(kl, "axis", -1)
            canonical["layerAttributes"]["axis"] = ax[0] if isinstance(ax, (list, tuple)) else ax
            if hasattr(kl, "data_format"):
                canonical["channels_last"] = str(kl.data_format) == "channels_last"

    elif layer_type in ("Conv2D", "Conv1D"):
        canonical["layerWeight"] = [name + "_kernel", name + "_bias"]
        k_w, b_w = _get_conv_weights_from_keras(keras_model, name)
        if k_w is None:
            raise RuntimeError("Conv layer " + name + " needs keras_model= for kernel/bias weights")
        canonical["initialisers"][name + "_kernel"] = np.ascontiguousarray(k_w, dtype=np.float32)
        canonical["initialisers"][name + "_bias"] = np.ascontiguousarray(b_w.flatten(), dtype=np.float32)
        kl = _get_keras_layer(keras_model, name)
        if kl is not None:
            _fill_conv_attributes_from_keras(canonical, kl)

    elif layer_type in ("MaxPooling2D", "AveragePooling2D", "GlobalAveragePooling2D"):
        kl = _get_keras_layer(keras_model, name)
        if kl is not None:
            _fill_pool_attributes_from_keras(canonical, kl)

    elif layer_type in ("Add", "Subtract", "Multiply"):
        if len(inputs) < 2:
            return None

    elif layer_type == "Concatenate":
        canonical["layerAttributes"]["axis"] = canonical["layerAttributes"].get("axis", 1)

    elif layer_type in ("Reshape", "Flatten"):
        shape = cfg_layer.get("output_shape")
        if shape is not None:
            try:
                canonical["layerAttributes"]["target_shape"] = list(shape)
            except Exception:
                pass
        if not canonical["layerAttributes"].get("target_shape"):
            if hasattr(hls_layer, "get_output_variable"):
                try:
                    var = hls_layer.get_output_variable()
                    if hasattr(var, "shape"):
                        canonical["layerAttributes"]["target_shape"] = list(var.shape)
                except Exception:
                    pass

    elif layer_type == "Activation":
        attrs["activation"] = attrs.get("Activation", attrs.get("activation", ""))

    elif layer_type == "LeakyReLU":
        kl = _get_keras_layer(keras_model, name)
        if kl is not None and hasattr(kl, "alpha"):
            canonical["layerAttributes"]["alpha"] = float(kl.alpha)

    if layer_type == "Input":
        pass

    return canonical


def extract_hls_config(hls_model: Any, keras_model: Any = None) -> Dict[str, Any]:
    # build canonical cfg from hls model
    name = getattr(hls_model, "name", None) or "hls_model"

    if hasattr(hls_model, "get_layers"):
        hls_layers = hls_model.get_layers()
    else:
        hls_layers = getattr(hls_model, "layers", [])

    raw_layers: List[Dict[str, Any]] = []
    for layer in hls_layers:
        layer_name = getattr(layer, "name", "")
        layer_type_name = type(layer).__name__

        attrs: Dict[str, Any] = {}
        if hasattr(layer, "attributes"):
            try:
                for k, v in layer.attributes.items():
                    key = str(k)
                    try:
                        attrs[key] = v
                    except Exception:
                        attrs[key] = str(v)
            except Exception:
                attrs = {}

        inputs: List[str] = []
        if hasattr(layer, "inputs"):
            for x in layer.inputs:
                n = getattr(x, "name", None)
                inputs.append(str(x) if n is None else n)

        outputs: List[str] = []
        if hasattr(layer, "outputs"):
            for x in layer.outputs:
                n = getattr(x, "name", None)
                outputs.append(str(x) if n is None else n)

        shape = None
        if hasattr(layer, "get_output_variable"):
            try:
                var = layer.get_output_variable()
                if hasattr(var, "shape"):
                    try:
                        shape = list(var.shape)
                    except Exception:
                        shape = None
            except Exception:
                shape = None

        raw_layers.append(
            {
                "name": layer_name,
                "class_name": layer_type_name,
                "attributes": attrs,
                "inputs": inputs,
                "outputs": outputs,
                "output_shape": shape,
            }
        )

    tensor_shapes: Dict[str, List[int]] = {}
    if hasattr(hls_model, "inputs"):
        for inp in hls_model.inputs:
            n = getattr(inp, "name", None)
            nm = str(inp) if n is None else n
            if hasattr(inp, "shape"):
                try:
                    tensor_shapes[nm] = list(inp.shape)
                except Exception:
                    pass
    for rl in raw_layers:
        sh = rl.get("output_shape")
        if sh:
            for on in rl.get("outputs", []):
                tensor_shapes[on] = list(sh)

    canonical_layers: List[Dict[str, Any]] = []
    for hls_layer, cfg_layer in zip(hls_layers, raw_layers):
        try:
            canon = _canonicalize_layer(hls_layer, cfg_layer, keras_model, tensor_shapes)
        except RuntimeError:
            raise
        except Exception:
            canon = None
        if canon is not None:
            canonical_layers.append(canon)

    weights_meta: Dict[str, Any] = {}
    for L in canonical_layers:
        for wname, wval in (L.get("initialisers") or {}).items():
            try:
                weights_meta[wname] = {"shape": list(np.asarray(wval).shape)}
            except Exception:
                weights_meta[wname] = {"shape": []}

    inputs: List[str] = []
    if hasattr(hls_model, "inputs"):
        for x in hls_model.inputs:
            n = getattr(x, "name", None)
            inputs.append(str(x) if n is None else n)

    outputs: List[str] = []
    if hasattr(hls_model, "outputs"):
        for x in hls_model.outputs:
            n = getattr(x, "name", None)
            outputs.append(str(x) if n is None else n)

    return {
        "name": name,
        "layers": canonical_layers,
        "weights": weights_meta,
        "inputs": inputs or ["input_0"],
        "outputs": outputs,
        "input_shape": _infer_input_shape(hls_model),
    }


def extract_layers(hls_model: Any, keras_model: Any = None) -> List[Dict[str, Any]]:
    # helper alias
    return extract_hls_config(hls_model, keras_model=keras_model)["layers"]
