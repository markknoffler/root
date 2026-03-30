import ROOT


def is_accurate(tensor_a, tensor_b, tolerance=1e-2):
    tensor_a = tensor_a.flatten()
    tensor_b = tensor_b.flatten()
    for i in range(len(tensor_a)):
        if abs(tensor_a[i] - tensor_b[i]) > tolerance:
            print(tensor_a[i], tensor_b[i])
            return False
    return True


def _get_pyhls4ml():
    # Touch TMVA once so the facade attaches PyKeras / PyHLS4ML when available.
    _ = ROOT.TMVA
    sofie = ROOT.TMVA.Experimental.SOFIE
    if hasattr(sofie, "PyHLS4ML"):
        return sofie.PyHLS4ML
    from ROOT._pythonization._tmva._sofie._parser._hls4ml.parser import PyHLS4ML

    return PyHLS4ML


def generate_and_test_inference_hls4ml(model_file_path: str, generated_header_file_dir: str = None, batch_size=1):
    import keras
    import numpy as np
    import tensorflow as tf
    import hls4ml

    print("Tensorflow version: ", tf.__version__)
    print("Keras version: ", keras.__version__)
    print("Numpy version:", np.__version__)

    model_name = model_file_path[model_file_path.rfind("/") + 1 :].removesuffix(".keras")
    if generated_header_file_dir is None:
        last_idx = model_file_path.rfind("/")
        generated_header_file_dir = "./" if last_idx == -1 else model_file_path[:last_idx]
    generated_header_file_path = generated_header_file_dir + "/" + model_name + ".hxx"

    keras_model = keras.models.load_model(model_file_path)

    hls_cfg = hls4ml.utils.config_from_keras_model(keras_model, granularity="name")
    if isinstance(hls_cfg, dict):
        hls_cfg.setdefault("Model", {})["Precision"] = "float"
        hls_cfg["Model"]["Strategy"] = "latency"
        for _layer_name, layer_cfg in hls_cfg.get("LayerName", {}).items():
            if isinstance(layer_cfg, dict):
                layer_cfg["Precision"] = "float"
        # Explicitly set IOType to avoid some Vivado backend defaults that might fail
        hls_cfg["Model"]["IOType"] = "io_parallel"

    hls_model = hls4ml.converters.convert_from_keras_model(keras_model, hls_config=hls_cfg, backend="Vivado")

    print(
        "Generating inference code for the hls4ml model from",
        model_file_path,
        "in the header",
        generated_header_file_path,
    )
    PyHLS4ML = _get_pyhls4ml()
    rmodel = PyHLS4ML.ParseFromModelGraph(
        hls_model,
        name=model_name,
        keras_model=keras_model,
    )
    SOFIE = ROOT.TMVA.Experimental.SOFIE
    rmodel.Generate(SOFIE.Options.kDefault)
    rmodel.OutputGenerated(generated_header_file_path)

    print(f"Compiling SOFIE model {model_name}")
    compile_status = ROOT.gInterpreter.Declare(f'#include "{generated_header_file_path}"')
    if not compile_status:
        raise AssertionError(f"Error compiling header file {generated_header_file_path}")
    sofie_model_namespace = getattr(ROOT, "TMVA_SOFIE_" + model_name)
    inference_session = sofie_model_namespace.Session(generated_header_file_path.removesuffix(".hxx") + ".dat")

    input_tensors = []
    for model_input in keras_model.inputs:
        input_shape = list(model_input.shape)
        input_shape[0] = batch_size
        input_tensors.append(np.ones(input_shape, dtype="float32"))

    sofie_inference_result = inference_session.infer(*input_tensors)
    sofie_output_tensor_shape = list(rmodel.GetTensorShape(rmodel.GetOutputTensorNames()[0]))
    keras_inputs = input_tensors[0] if len(input_tensors) == 1 else input_tensors
    # Use predict() to avoid Keras input-structure warnings affecting comparisons.
    keras_inference_result = keras_model.predict(keras_inputs, verbose=0)
    if sofie_output_tensor_shape != list(keras_inference_result.shape):
        raise AssertionError("Output tensor dimensions from SOFIE and Keras do not match")

    sofie_inference_result = np.asarray(sofie_inference_result)
    keras_inference_result = np.asarray(keras_inference_result)
    if not is_accurate(sofie_inference_result, keras_inference_result):
        raise AssertionError("Inference results from SOFIE and Keras do not match")
