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
    wlist: Optional[List[Any]] = None
    if hasattr(hls_layer, "get_weights"):
        try:
            w = hls_layer.get_weights()
            if isinstance(w, dict):
                wdict = w
            elif isinstance(w, (list, tuple)):
                wlist = list(w)
            else:
                # Some hls4ml layers return `dict.values()` / ValuesView.
                try:
                    if hasattr(w, "__iter__"):
                        wlist = list(w)
                except Exception:
                    pass
        except Exception:
            pass
    if hasattr(hls_layer, "weights") and not wdict:
        try:
            wattr = hls_layer.weights
            if hasattr(wattr, "items"):
                wdict = dict(wattr)
        except Exception:
            pass

    out_keys = list(want.keys())
    out: Dict[str, Optional[np.ndarray]] = {k: None for k in out_keys}

    # If get_weights returned a positional list/tuple, map it by out_keys order.
    if wlist is not None:
        for i, out_key in enumerate(out_keys):
            if i >= len(wlist):
                break
            arr = _to_numpy(wlist[i])
            if arr is not None:
                out[out_key] = np.asarray(arr, dtype=np.float32)

    # If we already got something from positional weights, skip dict matching.
    if all(v is not None for v in out.values()):
        return out
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
    import os
    debug = os.environ.get("SOFIE_HLS4ML_DEBUG", "0") == "1"
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
        in_name = inputs[0] if inputs else None
        out_name = outputs[0] if outputs else None
        in_dim = None
        out_dim = None
        if in_name and in_name in tensor_shapes:
            shp = tensor_shapes[in_name]
            in_dim = int(shp[-1]) if len(shp) >= 1 else None
        if out_name and out_name in tensor_shapes:
            shp = tensor_shapes[out_name]
            out_dim = int(shp[-1]) if len(shp) >= 1 else None

        # Bias default: must match the selected output dimension.
        if b_arr is None:
            b_arr = np.zeros(out_dim if out_dim is not None else w_arr.shape[1], dtype=np.float32)
        b_arr = np.asarray(b_arr, dtype=np.float32).flatten()

        # Orient kernel to [in_dim, out_dim] for SOFIE Gemm.
        if w_arr is not None and hasattr(w_arr, "shape") and w_arr.ndim == 2:
            keep_in, keep_out = int(w_arr.shape[0]), int(w_arr.shape[1])
            trans_in, trans_out = int(w_arr.shape[1]), int(w_arr.shape[0])

            # Prefer matching bias length / tensor_shapes dims.
            choose_transpose = False
            if out_dim is not None:
                if keep_out == out_dim and trans_out != out_dim:
                    choose_transpose = False
                elif trans_out == out_dim and keep_out != out_dim:
                    choose_transpose = True
            if b_arr is not None and b_arr.shape[0] in (keep_out, trans_out):
                if b_arr.shape[0] == trans_out and b_arr.shape[0] != keep_out:
                    choose_transpose = True

            if choose_transpose:
                w_arr = w_arr.T.copy()

            # Update n_in/n_out based on final orientation.
            if w_arr.ndim == 2:
                canonical["layerAttributes"]["n_in"] = int(w_arr.shape[0])
                canonical["layerAttributes"]["n_out"] = int(w_arr.shape[1])

        canonical["initialisers"][name + "_W"] = np.ascontiguousarray(w_arr, dtype=np.float32)
        canonical["initialisers"][name + "_B"] = np.ascontiguousarray(b_arr.flatten(), dtype=np.float32)

    elif layer_type == "BatchNormalization":
        canonical["layerWeight"] = [name + "_scale", name + "_bias", name + "_mean", name + "_var"]
        in_name_for_bn = inputs[0] if inputs else None
        n_feat = None
        if in_name_for_bn and in_name_for_bn in tensor_shapes:
            shp = tensor_shapes[in_name_for_bn]
            if len(shp) >= 1:
                n_feat = int(shp[-1])
        if n_feat is None:
            try:
                n_feat = int(attrs.get("n_filt", 1))
            except Exception:
                n_feat = 1

        # Defaults keep the parser from crashing even if get_weights() shape/key
        # extraction is not perfectly aligned with our expectations.
        g = np.ones(n_feat, dtype=np.float32)
        b = np.zeros(n_feat, dtype=np.float32)
        m = np.zeros(n_feat, dtype=np.float32)
        v = np.ones(n_feat, dtype=np.float32)

        # Try to extract BN params from hls4ml layer weights.
        raw_w = None
        if hasattr(hls_layer, "get_weights"):
            try:
                raw_w = hls_layer.get_weights()
            except Exception:
                raw_w = None

        if isinstance(raw_w, (list, tuple)) and len(raw_w) >= 4:
            # Common ordering: [gamma, beta, mean, var]
            g_try = _to_numpy(raw_w[0])
            b_try = _to_numpy(raw_w[1])
            m_try = _to_numpy(raw_w[2])
            v_try = _to_numpy(raw_w[3])
            if g_try is not None and g_try.size > 0:
                g = np.asarray(g_try, dtype=np.float32).flatten()
            if b_try is not None and b_try.size > 0:
                b = np.asarray(b_try, dtype=np.float32).flatten()
            if m_try is not None and m_try.size > 0:
                m = np.asarray(m_try, dtype=np.float32).flatten()
            if v_try is not None and v_try.size > 0:
                v = np.asarray(v_try, dtype=np.float32).flatten()
        else:
            w = _weights_from_hls_by_key(
                hls_layer,
                {
                    "scale": ["scale", "gamma"],
                    "bias": ["bias", "beta"],
                    "mean": ["mean", "moving_mean"],
                    "var": ["var", "variance", "moving_variance"],
                    # Extra aliases some backends use
                    "scale2": ["gamma", "scale"],
                },
            )
            if w.get("scale") is not None:
                g = np.asarray(w["scale"], dtype=np.float32).flatten()
            if w.get("bias") is not None:
                b = np.asarray(w["bias"], dtype=np.float32).flatten()
            if w.get("mean") is not None:
                m = np.asarray(w["mean"], dtype=np.float32).flatten()
            if w.get("var") is not None:
                v = np.asarray(w["var"], dtype=np.float32).flatten()

        # Ensure final vectors have consistent length.
        def _fix_len(arr: np.ndarray, target: int, default_value: float) -> np.ndarray:
            arr = np.asarray(arr, dtype=np.float32).flatten()
            if arr.size == target:
                return arr
            if arr.size == 1:
                return np.full(target, float(arr[0]), dtype=np.float32)
            if target <= 0:
                return arr
            return np.full(target, default_value, dtype=np.float32)

        g = _fix_len(g, n_feat, 1.0)
        b = _fix_len(b, n_feat, 0.0)
        m = _fix_len(m, n_feat, 0.0)
        v = _fix_len(v, n_feat, 1.0)

        canonical["initialisers"][name + "_scale"] = np.ascontiguousarray(g, dtype=np.float32)
        canonical["initialisers"][name + "_bias"] = np.ascontiguousarray(b, dtype=np.float32)
        canonical["initialisers"][name + "_mean"] = np.ascontiguousarray(m, dtype=np.float32)
        canonical["initialisers"][name + "_var"] = np.ascontiguousarray(v, dtype=np.float32)

        # BatchNorm attributes
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
        in_name_for_conv = inputs[0] if inputs else None
        out_name_for_conv = outputs[0] if outputs else None

        expected_in_c = None
        if in_name_for_conv and in_name_for_conv in tensor_shapes:
            try:
                expected_in_c = int(tensor_shapes[in_name_for_conv][-1])
            except Exception:
                expected_in_c = None

        expected_out_c = None
        if "n_filt" in attrs:
            try:
                expected_out_c = int(attrs["n_filt"])
            except Exception:
                expected_out_c = None
        if expected_out_c is None and out_name_for_conv and out_name_for_conv in tensor_shapes:
            try:
                expected_out_c = int(tensor_shapes[out_name_for_conv][-1])
            except Exception:
                expected_out_c = None

        # Prefer deterministic extraction from raw hls weights by shape first,
        # then use key-based matching as fallback.
        k_w = None
        b_w = None
        raw_w = None
        if hasattr(hls_layer, "get_weights"):
            try:
                raw_w = hls_layer.get_weights()
            except Exception:
                raw_w = None

        raw_arrays: List[np.ndarray] = []
        if isinstance(raw_w, dict):
            it = list(raw_w.values())
        elif isinstance(raw_w, (list, tuple)):
            it = list(raw_w)
        else:
            # hls4ml Vivado backend can return `collections.abc.ValuesView`
            # (e.g. `dict.values()`). Treat any iterable as a sequence of weights.
            it = []
            try:
                if raw_w is not None and hasattr(raw_w, "__iter__"):
                    it = list(raw_w)
            except Exception:
                it = []
        for _v in it:
            arr = _to_numpy(_v)
            if arr is not None:
                raw_arrays.append(np.asarray(arr, dtype=np.float32))

        # Conv2D kernel is expected to be 4D, bias is usually 1D.
        if layer_type == "Conv2D":
            kernel_candidates: List[np.ndarray] = []
            bias_candidates: List[np.ndarray] = []
            for arr in raw_arrays:
                if getattr(arr, "ndim", 0) == 4:
                    kernel_candidates.append(arr)
                if getattr(arr, "ndim", 0) == 1:
                    bias_candidates.append(arr.flatten())

            # Choose bias candidate that matches expected output channels.
            if expected_out_c is not None:
                for arr in bias_candidates:
                    if int(arr.shape[0]) == int(expected_out_c):
                        b_w = arr
                        break
            if b_w is None and bias_candidates:
                b_w = bias_candidates[0]

            # Choose kernel candidate consistent with expected channel counts.
            if kernel_candidates:
                if expected_in_c is not None and expected_out_c is not None:
                    for arr in kernel_candidates:
                        shp = list(arr.shape)
                        if int(expected_out_c) in shp and int(expected_in_c) in shp:
                            k_w = arr
                            break
                if k_w is None:
                    k_w = kernel_candidates[0]

        if k_w is None or b_w is None:
            w = _weights_from_hls_by_key(
                hls_layer,
                {"kernel": ["kernel", "weight", "weights", "w"], "bias": ["bias", "b"]},
            )
            if k_w is None:
                k_w = w["kernel"]
            if b_w is None:
                b_w = w["bias"]

        if k_w is None:
            raise RuntimeError("Conv layer " + name + " has no kernel weights in hls4ml layer")

        in_c = expected_in_c
        out_c = None
        if b_w is not None:
            try:
                out_c = int(np.asarray(b_w, dtype=np.float32).flatten().shape[0])
            except Exception:
                out_c = None
        if out_c is None:
            out_c = expected_out_c

        if b_w is None:
            b_w = np.zeros(int(out_c if out_c is not None else attrs.get("n_filt", 1)), dtype=np.float32)

        # Normalize kernel layout to OIHW for SOFIE.
        if layer_type == "Conv2D" and getattr(k_w, "ndim", 0) == 4:
            kernel = np.asarray(k_w, dtype=np.float32)
            chosen_layout = "unknown"
            if in_c is not None and out_c is not None:
                # Prefer HWIO first; fall back to other common layouts.
                if int(kernel.shape[-2]) == in_c and int(kernel.shape[-1]) == out_c:
                    k_w = np.transpose(kernel, (3, 2, 0, 1)).copy()  # HWIO -> OIHW
                    chosen_layout = "HWIO"
                elif int(kernel.shape[0]) == out_c and int(kernel.shape[-1]) == in_c:
                    k_w = np.transpose(kernel, (0, 3, 1, 2)).copy()  # OHWI -> OIHW
                    chosen_layout = "OHWI"
                elif int(kernel.shape[0]) == out_c and int(kernel.shape[1]) == in_c:
                    k_w = kernel.copy()  # already OIHW
                    chosen_layout = "OIHW"
                elif int(kernel.shape[0]) == in_c and int(kernel.shape[1]) == out_c:
                    k_w = np.transpose(kernel, (1, 0, 2, 3)).copy()  # IOHW -> OIHW
                    chosen_layout = "IOHW"
                else:
                    # Last-resort exhaustive attempt.
                    for perm in ((0, 1, 2, 3), (3, 2, 0, 1), (0, 3, 1, 2), (1, 0, 2, 3), (3, 0, 1, 2), (2, 3, 0, 1)):
                        try:
                            t = np.transpose(kernel, perm)
                            if int(t.shape[0]) == out_c and int(t.shape[1]) == in_c:
                                k_w = t.copy()
                                chosen_layout = f"perm{perm}"
                                break
                        except Exception:
                            continue
            else:
                # If channel counts are unknown, assume HWIO (most common source layout).
                k_w = np.transpose(kernel, (3, 2, 0, 1)).copy()
                chosen_layout = "HWIO_no_channels"
            if debug:
                try:
                    print(
                        "DEBUG CONV EXTRACT "
                        f"name={name} raw_kernel_shape={list(kernel.shape)} "
                        f"raw_arrays={[list(a.shape) for a in raw_arrays]} "
                        f"expected_in_c={expected_in_c} expected_out_c={expected_out_c} "
                        f"bias_shape={list(np.asarray(b_w).flatten().shape) if b_w is not None else None} "
                        f"chosen_layout={chosen_layout} final_kernel_shape={list(np.asarray(k_w).shape)}"
                    )
                except Exception:
                    pass
        canonical["initialisers"][name + "_kernel"] = np.ascontiguousarray(k_w, dtype=np.float32)
        canonical["initialisers"][name + "_bias"] = np.ascontiguousarray(b_w.flatten(), dtype=np.float32)

        # Conv attributes: SOFIE handlers assume these keys exist.
        def _derive_padding(attrs_local: Dict[str, Any]) -> str:
            p = attrs_local.get("padding", None)
            if p is None:
                p = attrs_local.get("pad_mode", None)
            if p is None:
                p = attrs_local.get("auto_pad", None)
            if p is not None:
                ps = str(p).strip().lower()
                if "same" in ps:
                    return "same"
                if "valid" in ps:
                    return "valid"

            pad_keys = ("pad_top", "pad_bottom", "pad_left", "pad_right")
            for pk in pad_keys:
                if pk in attrs_local:
                    try:
                        if int(attrs_local[pk]) > 0:
                            return "same"
                    except Exception:
                        pass
            return "valid"

        canonical["layerAttributes"]["padding"] = _derive_padding(attrs)

        # Kernel size
        if "kernel_height" in attrs and "kernel_width" in attrs:
            canonical["layerAttributes"]["kernel_size"] = [int(attrs["kernel_height"]), int(attrs["kernel_width"])]
        else:
            canonical["layerAttributes"]["kernel_size"] = list(_coerce_tuple(attrs.get("kernel_size"), (1, 1)))

        # Strides
        if "stride_height" in attrs and "stride_width" in attrs:
            canonical["layerAttributes"]["strides"] = [int(attrs["stride_height"]), int(attrs["stride_width"])]
        else:
            canonical["layerAttributes"]["strides"] = list(_coerce_tuple(attrs.get("strides"), (1, 1)))

        canonical["layerAttributes"]["groups"] = int(attrs.get("groups", 1))
        canonical["layerAttributes"]["dilation_rate"] = list(_coerce_tuple(attrs.get("dilation_rate"), (1, 1)))

        if "n_filt" in attrs:
            canonical["layerAttributes"]["n_filt"] = int(attrs["n_filt"])
        elif hasattr(k_w, "shape") and getattr(k_w, "ndim", 0) == 4:
            # k_w is [out, in, kh, kw] after our transpose heuristic (or should be).
            try:
                canonical["layerAttributes"]["n_filt"] = int(k_w.shape[0])
            except Exception:
                pass

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
        # Extract alpha from hls4ml layer attributes (no keras dependency).
        alpha_val = attrs.get("alpha", None)
        if alpha_val is None:
            alpha_val = attrs.get("elu_alpha", attrs.get("Alpha", None))
        if alpha_val is not None:
            try:
                canonical["layerAttributes"]["alpha"] = float(alpha_val)
            except Exception:
                pass

    elif layer_type == "LeakyReLU":
        # Extract leaky slope from hls4ml layer attributes (no keras dependency).
        neg_slope = attrs.get("negative_slope", None)
        if neg_slope is None:
            neg_slope = attrs.get("alpha", attrs.get("Alpha", None))
        if neg_slope is not None:
            try:
                canonical["layerAttributes"]["negative_slope"] = float(neg_slope)
                canonical["layerAttributes"]["alpha"] = float(neg_slope)
            except Exception:
                pass

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
    # Always derive per-input shapes from the extracted `tensor_shapes` map.
    # This keeps the build stage independent from any Keras model.
    if hasattr(hls_model, "inputs"):
        for x in hls_model.inputs:
            k = _normalize_tensor_name(getattr(x, "name", None) or str(x))
            shp = tensor_shapes.get(k)
            if shp is None:
                continue
            input_node_shapes[k] = [int(v) if v and int(v) > 0 else 1 for v in shp]

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
