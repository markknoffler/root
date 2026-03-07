import os
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

try:
    import keras
    from keras.models import load_model
except ImportError:
    from tensorflow import keras
    from tensorflow.keras.models import load_model

import ROOT
ROOT.gSystem.Load("libROOTTMVASofie")
ROOT.gInterpreter.Declare('#include "TMVA/RSofieReader.hxx"')


MODEL_FILES = [
    "ex5_gru.keras",
    "ex5_lstm.keras",
    "ex5_conv_transpose_valid.keras",
    "ex5_conv_transpose_same.keras",
]


def check_files():
    missing = [f for f in MODEL_FILES if not os.path.exists(f)]
    if missing:
        print("missing model files:", missing)
        print("run exercise5_make_keras_models.py first")
        sys.exit(1)


def parse_and_generate(model_path, batch_size):
    import ROOT
    rmodel = ROOT.TMVA.Experimental.SOFIE.PyKeras.Parse(model_path, batch_size)
    rmodel.Generate(ROOT.TMVA.Experimental.SOFIE.Options.kDefault, batch_size, 0, 0)
    base = os.path.splitext(os.path.basename(model_path))[0]
    hxx_path = base + "_sofie.hxx"
    rmodel.OutputGenerated(hxx_path)
    return hxx_path


def sofie_infer(hxx_path, dat_path, x):
    import ROOT
    ROOT.gInterpreter.Declare(f'#include "{hxx_path}"')
    session_class = hxx_path.replace("_sofie.hxx", "").replace("-", "_")
    # the generated namespace is TMVA_SOFIE_<modelname>
    model_name = os.path.splitext(os.path.basename(hxx_path))[0].replace("_sofie", "")
    ns = "TMVA_SOFIE_" + model_name
    session = getattr(ROOT, ns).Session(dat_path)
    flat = x.flatten().astype("float32")
    result = session.infer(flat)
    return np.array(list(result))


def run_test(model_path, x, batch_size):
    print(f"\n=== {model_path} ===")

    keras_model = load_model(model_path)
    keras_out = keras_model.predict(x, verbose=0)
    print(f"  keras output shape: {keras_out.shape}")

    base = os.path.splitext(os.path.basename(model_path))[0]
    hxx_path = base + "_sofie.hxx"
    dat_path = base + "_sofie.dat"

    if not os.path.exists(hxx_path) or not os.path.exists(dat_path):
        print(f"  SKIPPED - header not found ({hxx_path}), run parse-only tests first")
        return

    sofie_out = sofie_infer(hxx_path, dat_path, x)
    print(f"  sofie output size:  {sofie_out.size}")

    n = keras_out.size
    diff = np.max(np.abs(keras_out.flatten() - sofie_out.flatten()[:n]))
    print(f"  max diff: {diff:.6f}", end="  ")

    if diff < 1e-4:
        print("PASSED")
    else:
        print("FAILED")


def run_parse_only_tests():
    import ROOT
    print("\n--- parse-only tests ---")
    configs = [
        ("ex5_gru.keras", 2),
        ("ex5_lstm.keras", 2),
        ("ex5_conv_transpose_valid.keras", 1),
        ("ex5_conv_transpose_same.keras", 1),
    ]
    for path, batch in configs:
        print(f"  {path} ...", end=" ")
        try:
            hxx = parse_and_generate(path, batch)
            print(f"OK -> {hxx}")
        except Exception as e:
            print(f"FAILED: {e}")


if __name__ == "__main__":
    check_files()

    run_parse_only_tests()

    print("\n--- inference tests ---")
    try:
        run_test("ex5_gru.keras",                    np.random.RandomState(0).rand(2, 5, 4).astype("float32"),   2)
        run_test("ex5_lstm.keras",                   np.random.RandomState(1).rand(2, 5, 4).astype("float32"),   2)
        run_test("ex5_conv_transpose_valid.keras",   np.random.RandomState(2).rand(1, 4, 4, 1).astype("float32"), 1)
        run_test("ex5_conv_transpose_same.keras",    np.random.RandomState(3).rand(1, 4, 4, 1).astype("float32"), 1)
    except Exception as e:
        print(f"\nerror during inference tests: {e}")

    print("\ndone.")
