import os
import sys

_repo = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _repo)

import subprocess
import tempfile

import hls4ml
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model

from tmva.hls_models.hls4ml_parser.config import extract_hls_config


def _add_conv_dilations(onnx_path):
    import onnx
    model = onnx.load(onnx_path)
    for node in model.graph.node:
        if node.op_type == "Conv":
            if not any(a.name == "dilations" for a in node.attribute):
                kernel_shape = None
                for a in node.attribute:
                    if a.name == "kernel_shape":
                        kernel_shape = list(a.ints)
                        break
                nd = len(kernel_shape) if kernel_shape else 2
                dilations = [1] * nd
                node.attribute.append(onnx.helper.make_attribute("dilations", dilations))
    fd, tmp_path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    try:
        onnx.save(model, tmp_path)
        return tmp_path
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _convert_to_channels_last(onnx_path):
    preprocessed = _add_conv_dilations(onnx_path)
    try:
        try:
            from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
        except ImportError:
            from qonnx.transformation.general import ConvertToChannelsLastAndClean
        model_wrapper = ModelWrapper(preprocessed)
        model_wrapper = cleanup_model(model_wrapper)
        model_wrapper = model_wrapper.transform(ConvertToChannelsLastAndClean())
        model_wrapper = cleanup_model(model_wrapper)
        return model_wrapper
    except ImportError:
        with tempfile.TemporaryDirectory() as td:
            clean1 = os.path.join(td, "clean1.onnx")
            cl = os.path.join(td, "channels_last.onnx")
            clean2 = os.path.join(td, "clean2.onnx")
            subprocess.run(["qonnx_clean", preprocessed, "-o", clean1], check=True)
            subprocess.run(["qonnx_to_channels_last", clean1, "-o", cl], check=True)
            subprocess.run(["qonnx_clean", cl, "-o", clean2], check=True)
            return ModelWrapper(clean2)
    finally:
        if os.path.exists(preprocessed):
            os.unlink(preprocessed)


def main():
    out_dir = os.path.join(_repo, "tmva", "sofie", "exercise_outputs")
    os.makedirs(out_dir, exist_ok=True)

    onnx_path = os.path.join(_repo, "tmva", "hls_models", "ConvWithAsymmetricPadding.onnx")
    model_wrapper = _convert_to_channels_last(onnx_path)
    config = hls4ml.utils.config_from_onnx_model(model_wrapper, granularity="name")
    hls_model = hls4ml.converters.convert_from_onnx_model(model_wrapper, hls_config=config)

    cfg = extract_hls_config(hls_model)

    out_path = os.path.join(out_dir, "exercise4_hls4ml_modelgraph_output.txt")
    out_json = os.path.join(out_dir, "exercise4_hls4ml_config.json")
    with open(out_path, "w") as f:
        f.write("hls4ml model type: {}\n".format(type(hls_model).__name__))
        if hasattr(hls_model, "get_layers"):
            f.write("hls4ml layers: {}\n".format([l.name for l in hls_model.get_layers()]))
        f.write("\n")
        f.write("hls4ml config keys: {}\n".format(list(cfg.keys())))
        f.write("model name: {}\n".format(cfg.get("name")))
        f.write("num layers: {}\n".format(len(cfg.get("layers", []))))
        f.write("layer names: {}\n".format([l.get("name") for l in cfg.get("layers", [])]))
        f.write("num weights: {}\n".format(len(cfg.get("weights", {}))))
        f.write("inputs: {}\n".format(cfg.get("inputs")))
        f.write("outputs: {}\n".format(cfg.get("outputs")))

    import json
    def _to_json_safe(v):
        if isinstance(v, (str, int, float, bool, type(None))):
            return v
        if isinstance(v, (list, tuple)):
            return [_to_json_safe(x) for x in v]
        if isinstance(v, dict):
            return {str(k): _to_json_safe(x) for k, x in v.items()}
        if hasattr(v, "shape"):
            return list(v)
        return str(v)
    try:
        cfg_ser = _to_json_safe(cfg)
        with open(out_json, "w") as fj:
            json.dump(cfg_ser, fj, indent=2)
        print("saved config (json):", out_json)
    except Exception:
        pass
    print("saved config (txt):", out_path)


if __name__ == "__main__":
    main()

