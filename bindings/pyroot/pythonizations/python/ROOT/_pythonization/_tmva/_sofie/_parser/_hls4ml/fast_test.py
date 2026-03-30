import os
import sys
import shutil
import numpy as np
import keras
from keras import layers, models
import hls4ml
import ROOT
import traceback

# --- Setup Environment ---
# Ensure we import the LOCAL parser.py and its components
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CUR_DIR)

import parser
import config
from schema import sofie_layer_dict

# Alias for our local PyHLS4ML
PyHLS4ML = parser.PyHLS4ML

# --- Test Utilities ---

def is_accurate(tensor_a, tensor_b, tolerance=1e-2):
    tensor_a = tensor_a.flatten()
    tensor_b = tensor_b.flatten()
    if len(tensor_a) != len(tensor_b):
        return False
    diff = np.abs(tensor_a - tensor_b)
    max_diff = np.max(diff)
    if max_diff > tolerance:
        print(f"Max difference: {max_diff}")
        return False
    return True

def generate_and_test(keras_model, model_name, batch_size=1):
    print(f"\n>>> Testing Model: {model_name}")
    
    # 1. Convert to HLS4ML
    hls_cfg = hls4ml.utils.config_from_keras_model(keras_model, granularity="name")
    if isinstance(hls_cfg, dict):
        hls_cfg.setdefault("Model", {})["Precision"] = "float"
        hls_cfg["Model"]["Strategy"] = "latency"
        hls_cfg["Model"]["IOType"] = "io_parallel"
        for _layer_name, layer_cfg in hls_cfg.get("LayerName", {}).items():
            if isinstance(layer_cfg, dict):
                layer_cfg["Precision"] = "float"

    hls_model = hls4ml.converters.convert_from_keras_model(keras_model, hls_config=hls_cfg)
    
    # 2. Parse using LOCAL parser
    rmodel = PyHLS4ML.ParseFromModelGraph(hls_model, name=model_name, keras_model=keras_model)
    
    # 3. Generate Code
    SOFIE = ROOT.TMVA.Experimental.SOFIE
    rmodel.Generate(SOFIE.Options.kDefault)
    header_path = f"{model_name}.hxx"
    rmodel.OutputGenerated(header_path)
    
    # 4. Compile and Load
    if not ROOT.gInterpreter.Declare(f'#include "{header_path}"'):
        raise AssertionError(f"Error compiling header file {header_path}")
    
    sofie_model_namespace = getattr(ROOT, "TMVA_SOFIE_" + model_name)
    instance_name = f"session_{model_name}"
    # Use the generated .dat file if it exists, or just initialize
    inference_session = sofie_model_namespace.Session(model_name + ".dat")
    
    # 5. Inference
    input_tensors = []
    if isinstance(keras_model.input_shape, list):
        for shape in keras_model.input_shape:
            s = [ (int(x) if x and x > 0 else batch_size) for x in shape ]
            input_tensors.append(np.random.rand(*s).astype("float32"))
    else:
        s = [ (int(x) if x and x > 0 else batch_size) for x in keras_model.input_shape ]
        input_tensors.append(np.random.rand(*s).astype("float32"))
        
    sofie_result = inference_session.infer(*input_tensors)
    
    keras_inputs = input_tensors[0] if len(input_tensors) == 1 else input_tensors
    keras_result = keras_model.predict(keras_inputs, verbose=0)
    
    # 6. Verify
    sofie_res_np = np.asarray(sofie_result)
    keras_res_np = np.asarray(keras_result)
    
    if not is_accurate(sofie_res_np, keras_res_np):
        print(f"FAILED accuracy for {model_name}")
        return False
    
    print(f"PASSED: {model_name}")
    return True

# --- Test Case Definitions ---

def get_test_models():
    test_cases = []
    
    # 1. Simple Dense
    m = models.Sequential([layers.Input(shape=(10,)), layers.Dense(16, activation='relu'), layers.Dense(2, activation='softmax')])
    test_cases.append((m, "Dense_Softmax"))
    
    # 2. Conv2D Padding Same (Channels Last)
    m = models.Sequential([layers.Input(shape=(16, 16, 3)), layers.Conv2D(8, (3, 3), padding='same', activation='relu')])
    test_cases.append((m, "Conv2D_Same"))
    
    # 3. BatchNorm
    m = models.Sequential([layers.Input(shape=(10,)), layers.Dense(10), layers.BatchNormalization(), layers.Activation('relu')])
    test_cases.append((m, "BatchNorm"))
    
    # 4. Pooling
    m = models.Sequential([layers.Input(shape=(16, 16, 1)), layers.MaxPooling2D((2, 2)), layers.Flatten(), layers.Dense(1)])
    test_cases.append((m, "Pooling_Flatten"))
    
    # 5. Functional Add
    i1 = layers.Input(shape=(5,))
    i2 = layers.Input(shape=(5,))
    o = layers.Add()([i1, i2])
    test_cases.append((models.Model([i1, i2], o), "Functional_Add"))
    
    return test_cases

# --- Main Runner ---

if __name__ == "__main__":
    print("Starting Comprehensive Fast Test...")
    test_models = get_test_models()
    results = []
    
    for model, name in test_models:
        try:
            success = generate_and_test(model, name)
            results.append((name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"CRASHED: {name} with error: {e}")
            traceback.print_exc()
            results.append((name, "CRASH"))
            
    print("\n--- Final Results Summary ---")
    for name, status in results:
        print(f"{name:25} : {status}")
    
    # Cleanup generated files
    for name, _ in test_models:
        for ext in [".hxx", ".dat"]:
            if os.path.exists(name + ext):
                os.remove(name + ext)
