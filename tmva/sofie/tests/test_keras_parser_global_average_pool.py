#!/usr/bin/env python3
"""
Test that reproduces the GloabalAveragePool typo bug in the Keras parser.

Bug: In pooling.py line 64, PoolOpMode.GloabalAveragePool (typo) is used instead of
     PoolOpMode.GlobalAveragePool. The C++ enum only defines GlobalAveragePool,
     so parsing a model with GlobalAveragePooling2D raises AttributeError.

This test creates a minimal Keras model with GlobalAveragePooling2D, parses it via
PyKeras.Parse, and asserts the parse succeeds. The test FAILS when the bug exists
(AttributeError: GloabalAveragePool) and PASSES when the typo is fixed.

Run from ROOT build directory or with PYTHONPATH including ROOT bindings:
  python tmva/sofie/tests/test_keras_parser_global_average_pool.py
"""
import os
import sys
import tempfile
import unittest

# Skip if keras or ROOT not available
try:
    from keras import layers, models
    import numpy as np
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False

try:
    import ROOT
    from ROOT.TMVA.Experimental import SOFIE
    HAS_ROOT = True
except (ImportError, AttributeError):
    HAS_ROOT = False


@unittest.skipIf(not HAS_KERAS, "Keras not available")
@unittest.skipIf(not HAS_ROOT, "ROOT with PyROOT not available")
class TestKerasParserGlobalAveragePool(unittest.TestCase):
    """
    Regression test for the GloabalAveragePool typo in pooling.py.
    """

    def test_global_average_pooling2d_parse_succeeds(self):
        """
        Parse a Keras model containing GlobalAveragePooling2D via PyKeras.Parse.
        Fails with AttributeError when pooling.py uses GloabalAveragePool (typo).
        Passes when the correct enum GlobalAveragePool is used.
        """
        # Create minimal model with GlobalAveragePooling2D (channels_last)
        model = models.Sequential([
            layers.Input(shape=(4, 6, 3)),
            layers.GlobalAveragePooling2D(data_format='channels_last')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train one step so the model has valid weights/structure
        x = np.random.rand(2, 4, 6, 3).astype('float32')
        y = np.random.rand(2, 3).astype('float32')
        model.fit(x, y, epochs=1, verbose=0)

        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as f:
            model_path = f.name

        try:
            model.save(model_path)

            # This line raises AttributeError when the typo exists:
            #   AttributeError: type object 'PoolOpMode' has no attribute 'GloabalAveragePool'
            rmodel = ROOT.TMVA.Experimental.SOFIE.PyKeras.Parse(model_path, batch_size=1)

            self.assertIsNotNone(rmodel)
            output_names = rmodel.GetOutputTensorNames()
            self.assertGreater(len(output_names), 0)

        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)


if __name__ == '__main__':
    unittest.main()
