def generate_hls4ml_functional(dst_dir):

    import numpy as np
    from keras import layers, models
    from parser_test_function import is_channels_first_supported

    def train_and_save(model, name):
        if isinstance(model.input_shape, list):
            x_train = [np.random.rand(32, *shape[1:]) for shape in model.input_shape]
        else:
            x_train = np.random.rand(32, *model.input_shape[1:])
        y_train = np.random.rand(32, *model.output_shape[1:])
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
        model.fit(x_train, y_train, epochs=1, verbose=0)
        model.save(f"{dst_dir}/Functional_{name}_test.keras")

    for act in ["relu", "elu", "leaky_relu", "selu", "sigmoid", "softmax", "swish", "tanh"]:
        inp = layers.Input(shape=(10,))
        out = layers.Activation(act)(inp)
        train_and_save(models.Model(inp, out), f"Activation_layer_{act.capitalize()}")

    in1 = layers.Input(shape=(8,))
    in2 = layers.Input(shape=(8,))
    train_and_save(models.Model([in1, in2], layers.Add()([in1, in2])), "Add")

    if is_channels_first_supported():
        inp = layers.Input(shape=(3, 8, 8))
        out = layers.AveragePooling2D(pool_size=(2, 2), data_format="channels_first")(inp)
        train_and_save(models.Model(inp, out), "AveragePooling2D_channels_first")

    inp = layers.Input(shape=(8, 8, 3))
    out = layers.AveragePooling2D(pool_size=(2, 2), data_format="channels_last")(inp)
    train_and_save(models.Model(inp, out), "AveragePooling2D_channels_last")

    inp = layers.Input(shape=(10, 3, 5))
    out = layers.BatchNormalization(axis=2)(inp)
    train_and_save(models.Model(inp, out), "BatchNorm")

    in1 = layers.Input(shape=(8,))
    in2 = layers.Input(shape=(8,))
    train_and_save(models.Model([in1, in2], layers.Concatenate()([in1, in2])), "Concat")

    if is_channels_first_supported():
        inp = layers.Input(shape=(3, 8, 8))
        out = layers.Conv2D(4, (3, 3), padding="same", data_format="channels_first", activation="relu")(inp)
        train_and_save(models.Model(inp, out), "Conv2D_channels_first")

    inp = layers.Input(shape=(8, 8, 3))
    out = layers.Conv2D(4, (3, 3), padding="same", data_format="channels_last", activation="leaky_relu")(inp)
    train_and_save(models.Model(inp, out), "Conv2D_channels_last")

    inp = layers.Input(shape=(8, 8, 3))
    train_and_save(models.Model(inp, layers.Conv2D(4, (3, 3), padding="same", data_format="channels_last")(inp)), "Conv2D_padding_same")

    inp = layers.Input(shape=(8, 8, 3))
    train_and_save(models.Model(inp, layers.Conv2D(4, (3, 3), padding="valid", data_format="channels_last")(inp)), "Conv2D_padding_valid")

    inp = layers.Input(shape=(10,))
    train_and_save(models.Model(inp, layers.Dense(5, activation="tanh")(inp)), "Dense")

    inp = layers.Input(shape=(10,))
    train_and_save(models.Model(inp, layers.ELU(alpha=0.5)(inp)), "ELU")

    inp = layers.Input(shape=(4, 5))
    train_and_save(models.Model(inp, layers.Flatten()(inp)), "Flatten")

    if is_channels_first_supported():
        inp = layers.Input(shape=(3, 4, 6))
        train_and_save(models.Model(inp, layers.GlobalAveragePooling2D(data_format="channels_first")(inp)), "GlobalAveragePooling2D_channels_first")

    inp = layers.Input(shape=(4, 6, 3))
    train_and_save(models.Model(inp, layers.GlobalAveragePooling2D(data_format="channels_last")(inp)), "GlobalAveragePooling2D_channels_last")

    inp = layers.Input(shape=(10,))
    train_and_save(models.Model(inp, layers.LeakyReLU()(inp)), "LeakyReLU")

    if is_channels_first_supported():
        inp = layers.Input(shape=(3, 8, 8))
        out = layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(inp)
        train_and_save(models.Model(inp, out), "MaxPool2D_channels_first")

    inp = layers.Input(shape=(8, 8, 3))
    out = layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(inp)
    train_and_save(models.Model(inp, out), "MaxPool2D_channels_last")

    in1 = layers.Input(shape=(8,))
    in2 = layers.Input(shape=(8,))
    train_and_save(models.Model([in1, in2], layers.Multiply()([in1, in2])), "Multiply")

    inp = layers.Input(shape=(10,))
    train_and_save(models.Model(inp, layers.ReLU()(inp)), "ReLU")

    inp = layers.Input(shape=(4, 5))
    train_and_save(models.Model(inp, layers.Reshape((2, 10))(inp)), "Reshape")

    inp = layers.Input(shape=(10,))
    train_and_save(models.Model(inp, layers.Softmax()(inp)), "Softmax")

    in1 = layers.Input(shape=(8,))
    in2 = layers.Input(shape=(8,))
    train_and_save(models.Model([in1, in2], layers.Subtract()([in1, in2])), "Subtract")

    inp = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(8, (3, 3), padding="same", activation="relu", data_format="channels_last")(inp)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    out = layers.Dense(10, activation="softmax")(x)
    train_and_save(models.Model(inp, out), "Layer_Combination_1")

    inp = layers.Input(shape=(20,))
    x = layers.Dense(32, activation="tanh")(inp)
    x = layers.Dense(16)(x)
    x = layers.ELU()(x)
    out = layers.Dense(5, activation="sigmoid")(x)
    train_and_save(models.Model(inp, out), "Layer_Combination_2")

    inp1 = layers.Input(shape=(16,))
    inp2 = layers.Input(shape=(16,))
    d1 = layers.Dense(16, activation="relu")(inp1)
    d2 = layers.Dense(16, activation="selu")(inp2)
    add = layers.Add()([d1, d2])
    sub = layers.Subtract()([d1, d2])
    mul = layers.Multiply()([d1, d2])
    merged = layers.Concatenate()([add, sub, mul])
    merged = layers.LeakyReLU(alpha=0.1)(merged)
    out = layers.Dense(4, activation="softmax")(merged)
    train_and_save(models.Model([inp1, inp2], out), "Layer_Combination_3")
