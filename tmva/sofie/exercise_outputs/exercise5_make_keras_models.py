import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

try:
    import keras
    from keras.layers import GRU, LSTM, Conv2DTranspose
    from keras.models import Sequential
except ImportError:
    from tensorflow import keras
    from tensorflow.keras.layers import GRU, LSTM, Conv2DTranspose
    from tensorflow.keras.models import Sequential

print("Keras version:", keras.__version__)

rng = np.random.RandomState(42)


def save_model(model, name):
    path = name + ".keras"
    model.save(path)
    print("saved:", path)


def build_gru_model():
    model = Sequential()
    model.add(GRU(8, input_shape=(5, 4)))
    x = rng.rand(2, 5, 4).astype("float32")
    y = rng.rand(2, 8).astype("float32")
    model.compile(loss="mse", optimizer="adam")
    model.fit(x, y, epochs=2, verbose=0, batch_size=2)
    return model


def build_lstm_model():
    model = Sequential()
    model.add(LSTM(8, input_shape=(5, 4)))
    x = rng.rand(2, 5, 4).astype("float32")
    y = rng.rand(2, 8).astype("float32")
    model.compile(loss="mse", optimizer="adam")
    model.fit(x, y, epochs=2, verbose=0, batch_size=2)
    return model


def build_conv_transpose_valid():
    model = Sequential()
    model.add(Conv2DTranspose(4, kernel_size=3, strides=2, padding="valid", input_shape=(4, 4, 1)))
    x = rng.rand(1, 4, 4, 1).astype("float32")
    model.compile(loss="mse", optimizer="adam")
    model(x)
    return model


def build_conv_transpose_same():
    model = Sequential()
    model.add(Conv2DTranspose(4, kernel_size=3, strides=2, padding="same", input_shape=(4, 4, 1)))
    x = rng.rand(1, 4, 4, 1).astype("float32")
    model.compile(loss="mse", optimizer="adam")
    model(x)
    return model


print("\nbuilding GRU model...")
save_model(build_gru_model(), "ex5_gru")

print("\nbuilding LSTM model...")
save_model(build_lstm_model(), "ex5_lstm")

print("\nbuilding Conv2DTranspose (valid) model...")
save_model(build_conv_transpose_valid(), "ex5_conv_transpose_valid")

print("\nbuilding Conv2DTranspose (same) model...")
save_model(build_conv_transpose_same(), "ex5_conv_transpose_same")

print("\ndone.")
