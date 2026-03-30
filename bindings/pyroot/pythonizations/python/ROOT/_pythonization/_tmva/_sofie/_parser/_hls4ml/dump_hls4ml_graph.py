#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Any, Dict, List


def _norm_name(s: Any) -> str:
    # hls4ml/SOFIE tensor names must match exactly; still, for debugging,
    # we show a whitespace-free normalized version too.
    import re

    return re.sub(r"\s+", "", str(s))


def _var_name(v: Any) -> str:
    n = getattr(v, "name", None)
    if n is None:
        return str(v)
    return str(n)


def _var_shape(v: Any):
    return getattr(v, "shape", None)


def _load_local_parser():
    # Make sure we import our local hls4ml parser/config.
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)

    from config import extract_hls_config, _layer_type_from_hls

    return extract_hls_config, _layer_type_from_hls


def _to_attrs_dict(layer: Any) -> Dict[str, Any]:
    attrs = {}
    if hasattr(layer, "attributes"):
        try:
            for k, v in layer.attributes.items():
                try:
                    attrs[str(k)] = v
                except Exception:
                    attrs[str(k)] = str(v)
        except Exception:
            pass
    return attrs


def _get_layers(hls_model: Any) -> List[Any]:
    if hasattr(hls_model, "get_layers"):
        return hls_model.get_layers()
    return getattr(hls_model, "layers", [])


def main():
    ap = argparse.ArgumentParser(description="Dump hls4ml graph layers and inferred SOFIE schema layer types.")
    ap.add_argument(
        "--keras-model",
        default=None,
        help="Optional path to a saved Keras .keras model. If omitted, --test-model is used instead.",
    )
    ap.add_argument(
        "--test-model",
        default="Pooling_Flatten",
        help="Which built-in test model to construct if --keras-model is not provided.",
        choices=["Dense_Softmax", "Conv2D_Same", "BatchNorm", "Pooling_Flatten", "Functional_Add", "all"],
    )
    ap.add_argument("--backend", default="Vivado", help="hls4ml backend")
    ap.add_argument("--granularity", default="name", help="hls4ml config granularity")
    ap.add_argument("--io-type", default="io_parallel", help="hls4ml Model IOType (default io_parallel)")
    ap.add_argument("--strategy", default="latency", help="hls4ml Model Strategy")
    ap.add_argument("--precision", default="float", help="Set Precision to float for easier debugging")
    ap.add_argument("--max-layers", type=int, default=2000, help="Safety cap on dumped layers")
    args = ap.parse_args()

    import keras
    import numpy as np
    import hls4ml

    extract_hls_config, _layer_type_from_hls = _load_local_parser()

    def build_test_model(test_name: str):
        from keras import layers, models

        if test_name == "Dense_Softmax":
            return models.Sequential([layers.Input(shape=(10,)), layers.Dense(16, activation="relu"), layers.Dense(2, activation="softmax")])
        if test_name == "Conv2D_Same":
            return models.Sequential([layers.Input(shape=(16, 16, 3)), layers.Conv2D(8, (3, 3), padding="same", activation="relu")])
        if test_name == "BatchNorm":
            return models.Sequential([layers.Input(shape=(10,)), layers.Dense(10), layers.BatchNormalization(), layers.Activation("relu")])
        if test_name == "Pooling_Flatten":
            return models.Sequential([layers.Input(shape=(16, 16, 1)), layers.MaxPooling2D((2, 2)), layers.Flatten(), layers.Dense(1)])
        if test_name == "Functional_Add":
            i1 = layers.Input(shape=(5,))
            i2 = layers.Input(shape=(5,))
            o = layers.Add()([i1, i2])
            return models.Model([i1, i2], o)
        raise ValueError(f"Unknown test model {test_name}")

    test_names: List[str]
    if args.keras_model is not None:
        test_names = ["custom"]
    else:
        if args.test_model == "all":
            test_names = ["Dense_Softmax", "Conv2D_Same", "BatchNorm", "Pooling_Flatten", "Functional_Add"]
        else:
            test_names = [args.test_model]

    for tn in test_names:
        if tn == "custom":
            keras_model = keras.models.load_model(args.keras_model)
            model_label = os.path.basename(args.keras_model).removesuffix(".keras")
        else:
            keras_model = build_test_model(tn)
            model_label = tn

        hls_cfg = hls4ml.utils.config_from_keras_model(keras_model, granularity=args.granularity)
        if isinstance(hls_cfg, dict):
            hls_cfg.setdefault("Model", {})
            hls_cfg["Model"]["Precision"] = args.precision
            hls_cfg["Model"]["Strategy"] = args.strategy
            hls_cfg["Model"]["IOType"] = args.io_type
            for _, layer_cfg in hls_cfg.get("LayerName", {}).items():
                if isinstance(layer_cfg, dict):
                    layer_cfg["Precision"] = args.precision

        hls_model = hls4ml.converters.convert_from_keras_model(
            keras_model,
            hls_config=hls_cfg,
            backend=args.backend,
        )

        print(f"\n\n================ DUMP: {model_label} ================")

        print("===== Keras model =====")
        if args.keras_model is not None:
            print(f"path: {args.keras_model}")
        print(f"class: {type(keras_model).__name__}")
        print(f"num layers (keras): {len(getattr(keras_model, 'layers', []))}")

        print("\n===== hls4ml model IO =====")
        if hasattr(hls_model, "inputs"):
            for i, v in enumerate(hls_model.inputs):
                print(f"input[{i}]: name={_var_name(v)!r} norm={_norm_name(_var_name(v))!r} shape={_var_shape(v)}")
        if hasattr(hls_model, "outputs"):
            for i, v in enumerate(hls_model.outputs):
                print(f"output[{i}]: name={_var_name(v)!r} norm={_norm_name(_var_name(v))!r} shape={_var_shape(v)}")

        raw_layers = _get_layers(hls_model)
        print(f"\n===== hls4ml layers: {len(raw_layers)} total =====")

        # Dump raw layer info + inferred layer type according to our mapping function.
        dump_count = 0
        interesting = ("pool", "maxpool", "avgpool", "add", "subtract", "multiply", "flatten", "reshape")
        for idx, layer in enumerate(raw_layers):
            if dump_count >= args.max_layers:
                break

            class_name = type(layer).__name__
            layer_name = getattr(layer, "name", "")
            attrs = _to_attrs_dict(layer)

            # input/output tensor names
            in_names = []
            if hasattr(layer, "inputs"):
                for x in layer.inputs:
                    nm = _var_name(x)
                    in_names.append((nm, _norm_name(nm), _var_shape(x)))
            out_names = []
            if hasattr(layer, "outputs"):
                for x in layer.outputs:
                    nm = _var_name(x)
                    out_names.append((nm, _norm_name(nm), _var_shape(x)))

            # output variable shape if available
            out_shape = None
            if hasattr(layer, "get_output_variable"):
                try:
                    var = layer.get_output_variable()
                    out_shape = getattr(var, "shape", None)
                except Exception:
                    out_shape = None

            cfg_layer = {"class_name": class_name, "attributes": attrs}
            inferred = _layer_type_from_hls(layer, cfg_layer)

            cls_lc = str(class_name).lower()
            name_lc = str(layer_name).lower()
            is_interesting = any(k in cls_lc or k in name_lc for k in interesting)
            if not is_interesting and idx > 0 and idx % 50 != 0:
                continue

            dump_count += 1
            print(f"\n[hls_layer {idx}] class={class_name!r} name={layer_name!r} inferred={inferred!r}")
            if in_names:
                print("  inputs:")
                for j, (nm, nnm, shp) in enumerate(in_names):
                    print(f"    in[{j}]: name={nm!r} norm={nnm!r} shape={shp}")
            if out_names:
                print("  outputs:")
                for j, (nm, nnm, shp) in enumerate(out_names):
                    print(f"    out[{j}]: name={nm!r} norm={nnm!r} shape={shp}")
            if out_shape is not None:
                print(f"  out_shape(from get_output_variable)={out_shape}")
            if attrs:
                keys = sorted(list(attrs.keys()))
                preview_keys = [
                    k
                    for k in keys
                    if any(
                        t in k.lower()
                        for t in (
                            "pool",
                            "kernel",
                            "stride",
                            "padding",
                            "axis",
                            "activation",
                            "alpha",
                            "negative",
                            "dims",
                            "epsilon",
                            "groups",
                            "n_filt",
                        )
                    )
                ]
                if not preview_keys:
                    preview_keys = keys[:10]
                print("  attrs preview:")
                for k in preview_keys:
                    print(f"    {k}={attrs.get(k)!r}")

        # Dump canonical schema summary for just the problematic ops.
        print("\n===== CanonicalOpSchema (filtered) =====")
        cfg = extract_hls_config(hls_model, keras_model=keras_model)
        canonical_layers = cfg.get("layers", [])
        wanted = {
            "MaxPooling2D",
            "AveragePooling2D",
            "GlobalAveragePooling2D",
            "Add",
            "Subtract",
            "Multiply",
            "Flatten",
            "Reshape",
        }
        for cl in canonical_layers:
            lt = cl.get("layerType")
            if lt in wanted:
                print(
                    f"canonical: layerType={lt!r} name={cl.get('name')!r} "
                    f"in={cl.get('layerInput')!r} out={cl.get('layerOutput')!r} "
                    f"attrs_keys={list((cl.get('layerAttributes') or {}).keys())[:8]!r}"
                )


if __name__ == "__main__":
    main()

