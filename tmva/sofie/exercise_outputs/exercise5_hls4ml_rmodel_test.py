import os
import sys

_repo = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _repo)

import numpy as np
import ROOT
ROOT.gSystem.Load("libROOTTMVASofie")
from ROOT.TMVA.Experimental import SOFIE

import keras
from keras import layers
import hls4ml

from tmva.hls_models.hls4ml_parser.parser import PyHLS4ML


def build_keras_model():
    model = keras.Sequential(
        [
            layers.Dense(16, input_shape=(8,), activation="relu"),
            layers.Dense(8, activation="elu"),
            layers.Dense(4),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def main():
    out_dir = os.path.join(_repo, "tmva", "sofie", "exercise_outputs")
    os.makedirs(out_dir, exist_ok=True)

    model = build_keras_model()
    config = hls4ml.utils.config_from_keras_model(model, granularity="name")
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)

    print("hls4ml model type:", type(hls_model).__name__)
    if hasattr(hls_model, "get_layers"):
        print("hls4ml layers:", [l.name for l in hls_model.get_layers()])

    rmodel = PyHLS4ML.ParseFromModelGraph(hls_model, name="HLS4MLDenseModel", keras_model=model)

    rmodel.Generate()
    header_path = os.path.join(out_dir, "HLS4MLDenseModel_sofie.hxx")
    data_path = os.path.join(out_dir, "HLS4MLDenseModel_sofie.dat")
    rmodel.OutputGenerated(header_path)
    rmodel.OutputTensorFile(data_path)

    print("generated header:", header_path)
    print("generated data:", data_path)

    ROOT.gInterpreter.Declare('#include "' + header_path + '"')
    session = getattr(ROOT, "TMVA_SOFIE_HLS4MLDenseModel").Session(data_path)
    x = np.random.randn(1, 8).astype(np.float32)
    y_sofie = session.infer(x)
    y_keras = model(x)
    diff = np.abs(np.array(y_sofie).flatten() - np.array(y_keras).flatten())
    print("inference (sample input): SOFIE output =", list(y_sofie))
    print("inference (same input):   Keras output =", y_keras.numpy().tolist())
    print("max abs diff:", float(np.max(diff)))


if __name__ == "__main__":
    main()

