import unittest
import os
import shutil

from parser_test_function_hls4ml import generate_and_test_inference_hls4ml
from generate_hls4ml_functional import generate_hls4ml_functional
from generate_hls4ml_sequential import generate_hls4ml_sequential


def make_testname(test_case: str):
    test_case_name = test_case.replace("_", " ").removesuffix(".keras")
    return test_case_name


# One layer first; extend here as more hls4ml → SOFIE paths are fixed.
models = ["Dense"]


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
        self.run_model_tests("sequential", generate_hls4ml_sequential, models)

    def test_functional(self):
        self.run_model_tests("functional", generate_hls4ml_functional, models)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("sequential", ignore_errors=True)
        shutil.rmtree("functional", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
