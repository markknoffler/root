import ROOT


def is_accurate(tensor_a, tensor_b, tolerance=1e-2):
    tensor_a = tensor_a.flatten()
    tensor_b = tensor_b.flatten()
    for i in range(len(tensor_a)):
        if abs(tensor_a[i] - tensor_b[i]) > tolerance:
            print(tensor_a[i], tensor_b[i])
            return False
    return True


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
    keras_model.load_weights(model_file_path)

    hls_cfg = hls4ml.utils.config_from_keras_model(keras_model, granularity="name")
    hls_model = hls4ml.converters.convert_from_keras_model(keras_model, hls_config=hls_cfg)

    print(
        "Generating inference code for the hls4ml model from",
        model_file_path,
        "in the header",
        generated_header_file_path,
    )
    rmodel = ROOT.TMVA.Experimental.SOFIE.PyHLS4ML.ParseFromModelGraph(
        hls_model,
        name=model_name,
        keras_model=keras_model,
    )
    rmodel.Generate()
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
    keras_inference_result = keras_model(input_tensors)
    if sofie_output_tensor_shape != list(keras_inference_result.shape):
        raise AssertionError("Output tensor dimensions from SOFIE and Keras do not match")

    sofie_inference_result = np.asarray(sofie_inference_result)
    keras_inference_result = np.asarray(keras_inference_result)
    if not is_accurate(sofie_inference_result, keras_inference_result):
        raise AssertionError("Inference results from SOFIE and Keras do not match")
