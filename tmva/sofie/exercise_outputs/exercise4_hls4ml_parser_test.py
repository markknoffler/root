import os
import sys

_repo = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _repo)

import hls4ml
import onnx

from tmva.hls_models.hls4ml_parser.config import extract_hls_config


def main():
    out_dir = os.path.join(_repo, "tmva", "sofie", "exercise_outputs")
    os.makedirs(out_dir, exist_ok=True)

    onnx_path = os.path.join(_repo, "tmva", "hls_models", "ConvWithAsymmetricPadding.onnx")
    model = onnx.load(onnx_path)
    hls_model = hls4ml.converters.convert_from_onnx_model(model)

    cfg = extract_hls_config(hls_model)

    out_path = os.path.join(out_dir, "exercise4_hls4ml_modelgraph_output.txt")
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

    print("saved config to:", out_path)


if __name__ == "__main__":
    main()

