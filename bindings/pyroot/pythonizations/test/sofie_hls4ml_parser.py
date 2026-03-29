import unittest
import os
import shutil


from parser_test_function import is_channels_first_supported
from parser_test_function_hls4ml import generate_and_test_inference_hls4ml
from generate_hls4ml_functional import generate_hls4ml_functional
from generate_hls4ml_sequential import generate_hls4ml_sequential


def make_testname(test_case: str):
    test_case_name = test_case.replace("_", " ").removesuffix(".keras")
    return test_case_name


models = [
    "AveragePooling2D_channels_first",
    "AveragePooling2D_channels_last",
    "BatchNorm",
    "Conv2D_channels_first",
    "Conv2D_channels_last",
    "Conv2D_padding_same",
    "Conv2D_padding_valid",
    "Dense",
    "ELU",
    "Flatten",
    "GlobalAveragePooling2D_channels_first",
    "GlobalAveragePooling2D_channels_last",
    "LeakyReLU",
    "MaxPool2D_channels_first",
    "MaxPool2D_channels_last",
    "ReLU",
    "Reshape",
    "Softmax",
] + (
    [f"Activation_layer_{activation_function.capitalize()}" for activation_function in
     ["relu", "elu", "leaky_relu", "selu", "sigmoid", "softmax", "swish", "tanh"]] +
    [f"Layer_Combination_{i}" for i in range(1, 4)]
)

if not is_channels_first_supported():
    models = [m for m in models if "channels_first" not in m]

print(models)


class SOFIE_HLS4ML_Parser(unittest.TestCase):

    def setUp(self):
        base_dir = self._testMethodName[5:]
        if os.path.isdir(base_dir):
            shutil.rmtree(base_dir)
        os.makedirs(base_dir + "/input_models")
        os.makedirs(base_dir + "/generated_header_files_dir")

    def run_model_tests(self, model_type: str, generate_function, model_list):
        print("Generating", model_type, "models for testing")
        generate_function(f"{model_type}/input_models")
        for keras_model in model_list:
            print("**********************************")
            print("Run test for", model_type, "model:", keras_model)
            print("**********************************")
            keras_model_name = f"{model_type.capitalize()}_{keras_model}_test.keras"
            keras_model_path = f"{model_type}/input_models/" + keras_model_name
            with self.subTest(msg=make_testname(keras_model_name)):
                generate_and_test_inference_hls4ml(
                    keras_model_path,
                    f"{model_type}/generated_header_files_dir",
                )

    def test_sequential(self):
        sequential_models = models
        self.run_model_tests("sequential", generate_hls4ml_sequential, sequential_models)

    def test_functional(self):
        functional_models = models + ["Add", "Concat", "Multiply", "Subtract"]
        self.run_model_tests("functional", generate_hls4ml_functional, functional_models)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("sequential", ignore_errors=True)
        shutil.rmtree("functional", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
