from __future__ import annotations

import ast
import re
import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

def _normalize_tensor_name(name: Any) -> str:
    # SOFIE tensor names must match exactly. hls4ml sometimes returns names with
    # trailing spaces/newlines, so we canonicalize by removing all whitespace.
    import re
    return re.sub(r"\s+", "", str(name))


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

    # hls4ml Vivado backend layer class names.
    if class_name == "VivadoPooling2D":
        op = str(attrs.get("pool_op", "")).lower()
        if "max" in op:
            return "MaxPooling2D"
        if "avg" in op or "average" in op:
            return "AveragePooling2D"
        return "MaxPooling2D"
    if class_name == "VivadoMerge":
        fcpp = str(attrs.get("function_cpp", "")).lower()
        mod = str(attrs.get("module", "")).lower()
        if "nnet::add" in fcpp or ".layers.merging.add" in mod:
            return "Add"
        if "nnet::sub" in fcpp or ".layers.merging.subtract" in mod:
            return "Subtract"
        if "nnet::mul" in fcpp or "nnet::mult" in fcpp or ".layers.merging.multiply" in mod:
            return "Multiply"
        return None
    if class_name == "VivadoReshape":
        mod = str(attrs.get("module", "")).lower()
        nm = str(cfg_layer.get("name", getattr(hls_layer, "name", ""))).lower()
        if "flatten" in mod or nm == "flatten":
            return "Flatten"
        return "Reshape"
    lc = str(class_name).lower().replace("-", "_")
    # Pooling layer class names can appear as `MaxPooling2D`, `max_pooling2d`, `maxpool`, etc.
    if "maxpool" in lc or "max_pool" in lc or ("max" in lc and "pool" in lc):
        return "MaxPooling2D"
    if "averagepool" in lc or "average_pool" in lc or ("average" in lc and "pool" in lc) or ("avg" in lc and "pool" in lc):
        return "AveragePooling2D"
    if "globalaveragepool" in lc or "global_average_pool" in lc or ("global" in lc and "avg" in lc and "pool" in lc):
        return "GlobalAveragePooling2D"
    if lc in ("add", "subtract", "multiply"):
        return class_name.capitalize()
    if lc.startswith("add"):
        return "Add"
    if lc.startswith("subtract"):
        return "Subtract"
    if lc.startswith("multiply"):
        return "Multiply"
    if "Concat" in class_name:
        return "Concatenate"
    if "Reshape" in class_name:
        return "Reshape"
    if "Flatten" in class_name:
        return "Flatten"
    if "Permute" in class_name:
        return "Permute"
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


def _weights_from_hls_by_key(
    hls_layer: Any,
    want: Dict[str, List[str]],
) -> Dict[str, Optional[np.ndarray]]:
    """
    Extract named weight arrays from an hls4ml layer by heuristic key matching.
    `want` maps output field -> list of substrings that should appear in the weight key.
    """
    wdict: Dict[str, Any] = {}
    if hasattr(hls_layer, "get_weights"):
        try:
            w = hls_layer.get_weights()
            if isinstance(w, dict):
                wdict = w
        except Exception:
            pass
    if hasattr(hls_layer, "weights") and not wdict:
        try:
            wattr = hls_layer.weights
            if hasattr(wattr, "items"):
                wdict = dict(wattr)
        except Exception:
            pass

    out: Dict[str, Optional[np.ndarray]] = {k: None for k in want.keys()}
    for k, v in (wdict or {}).items():
        if v is None:
            continue
        ks = str(k).lower()
        arr = _to_numpy(v)
        if arr is None:
            continue
        for out_key, needles in want.items():
            if out[out_key] is not None:
                continue
            if any(n in ks for n in needles):
                out[out_key] = np.asarray(arr, dtype=np.float32)
    return out


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

    canonical = {
        "layerType": layer_type,
        "layerInput": inputs or [],
        "layerOutput": outputs or [],
        "layerDType": "float",
        "layerAttributes": copy.deepcopy(attrs),
        "layerWeight": [],
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
        if w_arr is None:
            raise RuntimeError(
                "Dense layer " + name + " has no weights in hls4ml layer"
            )
        if b_arr is None:
            b_arr = np.zeros(w_arr.shape[0], dtype=np.float32)
        canonical["initialisers"][name + "_W"] = np.ascontiguousarray(w_arr, dtype=np.float32)
        canonical["initialisers"][name + "_B"] = np.ascontiguousarray(b_arr.flatten(), dtype=np.float32)
        # SOFIE Dense output shape needs an explicit output dimension.
        # hls4ml extraction doesn't always provide this in layerAttributes.
        if w_arr is not None and hasattr(w_arr, "shape") and len(w_arr.shape) >= 2:
            canonical["layerAttributes"]["n_in"] = int(w_arr.shape[0])
            canonical["layerAttributes"]["n_out"] = int(w_arr.shape[1])

    elif layer_type == "BatchNormalization":
        canonical["layerWeight"] = [name + "_scale", name + "_bias", name + "_mean", name + "_var"]
        w = _weights_from_hls_by_key(
            hls_layer,
            {
                "scale": ["scale", "gamma"],
                "bias": ["bias", "beta"],
                "mean": ["mean", "moving_mean"],
                "var": ["var", "variance", "moving_variance"],
            },
        )
        g, b, m, v = w["scale"], w["bias"], w["mean"], w["var"]
        if g is None or b is None or m is None or v is None:
            raise RuntimeError("BatchNorm " + name + " has incomplete weights in hls4ml layer")
        canonical["initialisers"][name + "_scale"] = np.ascontiguousarray(g.flatten(), dtype=np.float32)
        canonical["initialisers"][name + "_bias"] = np.ascontiguousarray(b.flatten(), dtype=np.float32)
        canonical["initialisers"][name + "_mean"] = np.ascontiguousarray(m.flatten(), dtype=np.float32)
        canonical["initialisers"][name + "_var"] = np.ascontiguousarray(v.flatten(), dtype=np.float32)
        if "axis" in attrs:
            try:
                canonical["layerAttributes"]["axis"] = int(attrs.get("axis", -1))
            except Exception:
                pass
        if "epsilon" in attrs:
            try:
                canonical["layerAttributes"]["epsilon"] = float(attrs["epsilon"])
            except Exception:
                pass

    elif layer_type in ("Conv2D", "Conv1D"):
        canonical["layerWeight"] = [name + "_kernel", name + "_bias"]
        w = _weights_from_hls_by_key(
            hls_layer,
            {"kernel": ["kernel", "weight", "weights", "w"], "bias": ["bias", "b"]},
        )
        k_w, b_w = w["kernel"], w["bias"]
        if k_w is None:
            raise RuntimeError("Conv layer " + name + " has no kernel weights in hls4ml layer")
        if b_w is None:
            b_w = np.zeros(int(attrs.get("n_filt", k_w.shape[0])), dtype=np.float32)
        # Heuristic: if kernel looks like HWIO, transpose to OIHW for SOFIE.
        if k_w.ndim == 4 and (k_w.shape[-1] == int(attrs.get("n_filt", k_w.shape[-1]))):
            k_w = np.transpose(k_w, (3, 2, 0, 1)).copy()
        elif k_w.ndim == 4 and k_w.shape[0] != int(attrs.get("n_filt", k_w.shape[0])):
            k_w = np.transpose(k_w, (3, 2, 0, 1)).copy()
        canonical["initialisers"][name + "_kernel"] = np.ascontiguousarray(k_w, dtype=np.float32)
        canonical["initialisers"][name + "_bias"] = np.ascontiguousarray(b_w.flatten(), dtype=np.float32)
        if "padding" in attrs:
            canonical["layerAttributes"]["padding"] = str(attrs.get("padding", "valid"))
        if "stride_height" in attrs and "stride_width" in attrs:
            canonical["layerAttributes"]["strides"] = [int(attrs["stride_height"]), int(attrs["stride_width"])]
        if "strides" not in canonical["layerAttributes"]:
            canonical["layerAttributes"]["strides"] = list(_coerce_tuple(attrs.get("strides"), (1, 1)))
        if "kernel_size" not in canonical["layerAttributes"]:
            canonical["layerAttributes"]["kernel_size"] = list(_coerce_tuple(attrs.get("kernel_size"), (1, 1)))
        canonical["layerAttributes"]["groups"] = int(attrs.get("groups", 1))
        canonical["layerAttributes"]["dilation_rate"] = list(_coerce_tuple(attrs.get("dilation_rate"), (1, 1)))
        if "n_filt" in attrs:
            canonical["layerAttributes"]["n_filt"] = int(attrs["n_filt"])

    elif layer_type in ("MaxPooling2D", "AveragePooling2D", "GlobalAveragePooling2D"):
        ph = attrs.get("pool_height", None)
        pw = attrs.get("pool_width", None)
        sh = attrs.get("stride_height", None)
        sw = attrs.get("stride_width", None)
        if ph is not None and pw is not None:
            canonical["layerAttributes"]["pool_size"] = [int(ph), int(pw)]
        if sh is not None and sw is not None:
            canonical["layerAttributes"]["strides"] = [int(sh), int(sw)]
        canonical["layerAttributes"]["padding"] = str(attrs.get("padding", "valid"))

    elif layer_type in ("Add", "Subtract", "Multiply"):
        if len(inputs) < 2:
            return None

    elif layer_type == "Concatenate":
        canonical["layerAttributes"]["axis"] = canonical["layerAttributes"].get("axis", 1)

    elif layer_type in ("Reshape", "Flatten"):
        shape = cfg_layer.get("output_shape")
        if shape is not None:
            try:
                # SOFIE reshape mode expects shape excluding batch dim.
                shp_list = list(shape)
                if len(shp_list) >= 2:
                    shp_list = shp_list[1:]
                canonical["layerAttributes"]["target_shape"] = shp_list
            except Exception:
                pass
        if not canonical["layerAttributes"].get("target_shape"):
            if hasattr(hls_layer, "get_output_variable"):
                try:
                    var = hls_layer.get_output_variable()
                    if hasattr(var, "shape"):
                        shp_list = list(var.shape)
                        if len(shp_list) >= 2:
                            shp_list = shp_list[1:]
                        canonical["layerAttributes"]["target_shape"] = shp_list
                except Exception:
                    pass

    elif layer_type == "Permute":
        d = attrs.get("dims")
        if d is not None:
            canonical["layerAttributes"]["dims"] = list(_coerce_tuple(d, ()))
        else:
            # try to get from hls layer if missing in attrs
            if hasattr(hls_layer, "get_attr"):
                try:
                    canonical["layerAttributes"]["dims"] = list(hls_layer.get_attr("dims"))
                except Exception:
                    pass

    elif layer_type == "Activation":
        attrs["activation"] = attrs.get("Activation", attrs.get("activation", ""))

    elif layer_type == "ELU":
        kl = _get_keras_layer(keras_model, name)
        if kl is not None and hasattr(kl, "alpha"):
            canonical["layerAttributes"]["alpha"] = float(kl.alpha)

    elif layer_type == "LeakyReLU":
        kl = _get_keras_layer(keras_model, name)
        if kl is not None:
            if hasattr(kl, "alpha"):
                canonical["layerAttributes"]["alpha"] = float(kl.alpha)
            if hasattr(kl, "negative_slope"):
                canonical["layerAttributes"]["negative_slope"] = float(kl.negative_slope)
                # Keep compatibility with operators that only look at `alpha`.
                canonical["layerAttributes"]["alpha"] = float(kl.negative_slope)

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
                val = str(x) if n is None else str(n)
                inputs.append(_normalize_tensor_name(val))

        outputs: List[str] = []
        if hasattr(layer, "outputs"):
            for x in layer.outputs:
                n = getattr(x, "name", None)
                val = str(x) if n is None else str(n)
                outputs.append(_normalize_tensor_name(val))

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
            nm = _normalize_tensor_name(nm)
            if hasattr(inp, "shape"):
                try:
                    tensor_shapes[nm] = list(inp.shape)
                except Exception:
                    pass
    for rl in raw_layers:
        sh = rl.get("output_shape")
        if sh:
            for on in rl.get("outputs", []):
                tensor_shapes[str(on).strip()] = list(sh)

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
            val = str(x) if n is None else str(n)
            inputs.append(_normalize_tensor_name(val))

    outputs: List[str] = []
    if hasattr(hls_model, "outputs"):
        for x in hls_model.outputs:
            n = getattr(x, "name", None)
            val = str(x) if n is None else str(n)
            outputs.append(_normalize_tensor_name(val))

    # Provide explicit per-input shapes when we have a Keras model.
    # The build stage in SOFIE needs these exact ranks/dims; guessing from hls4ml inputs
    # can easily collapse to rank-1 shapes like [1] -> [1,1] and break conv/pool handlers.
    input_node_shapes: Dict[str, List[int]] = {}
    if keras_model is not None and hasattr(keras_model, "inputs") and hasattr(hls_model, "inputs"):
        for idx, x in enumerate(hls_model.inputs):
            if idx >= len(getattr(keras_model, "inputs", [])):
                break
            keras_in = keras_model.inputs[idx]
            shp = getattr(keras_in, "shape", None)
            if shp is None:
                continue
            shp_list = list(shp)
            # Drop batch dim (usually None) and replace with Keras feature shape.
            if len(shp_list) >= 1:
                shp_no_batch = shp_list[1:]
            else:
                shp_no_batch = shp_list
            if not shp_no_batch:
                shp_no_batch = [1]
            # Convert None/unknown dims to 1 for safe integer shapes.
            cleaned = []
            for d in shp_no_batch:
                try:
                    val = int(d)
                    cleaned.append(val if val > 0 else 1)
                except Exception:
                    cleaned.append(1)
            k = _normalize_tensor_name(getattr(x, "name", None) or str(x))
            input_node_shapes[k] = cleaned

    return {
        "name": name,
        "layers": canonical_layers,
        "weights": weights_meta,
        "inputs": inputs or ["input_0"],
        "outputs": outputs,
        "input_node_shapes": input_node_shapes,
        "input_shape": _infer_input_shape(hls_model),
        # Direct shape map from the hls4ml ModelGraph variables.
        # This is used by the build stage as a fallback for tensor registration.
        "tensor_shapes": tensor_shapes,
    }


def extract_layers(hls_model: Any, keras_model: Any = None) -> List[Dict[str, Any]]:
    # helper alias
    return extract_hls_config(hls_model, keras_model=keras_model)["layers"]
