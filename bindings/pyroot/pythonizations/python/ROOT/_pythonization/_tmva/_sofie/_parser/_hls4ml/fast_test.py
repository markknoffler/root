import os
import sys
import numpy as np
import keras
import hls4ml
import ROOT

# Add the current directory to sys.path to ensure we import the LOCAL parser.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parser
import config
from schema import sofie_layer_dict

def run_fast_inference():
    print("--- Starting Fast Inference Test ---")
    
    # 1. Create a simple model
    model = keras.Sequential([
        keras.layers.Input(shape=(10,)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # 2. Convert to HLS4ML
    hls_cfg = hls4ml.utils.config_from_keras_model(model, granularity='name')
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=hls_cfg)
    
    print("HLS4ML model converted.")
    
    # 3. Use our LOCAL parser to build SOFIE RModel
    print("Building SOFIE RModel using local parser...")
    try:
        # Use the static methods in PyHLS4ML class
        rmodel = parser.PyHLS4ML.ParseFromModelGraph(hls_model, name="FastModel", keras_model=model)
        print("RModel construction successful!")
        
        # 4. Briefly verify model properties
        outputs = list(rmodel.GetOutputTensorNames())
        print(f"Model Outputs: {outputs}")
        
        # 5. Try Generate (requires C++ backend, but we can check if it fails)
        SOFIE = ROOT.TMVA.Experimental.SOFIE
        rmodel.Generate(SOFIE.Options.kDefault)
        print("RModel code generation successful!")
        
        # 6. Final registration check
        rmodel.Initialize()
        print("RModel initialization successful!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_fast_inference()
