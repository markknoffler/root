def generate_hls4ml_sequential(dst_dir):

    import numpy as np
    from keras import layers, models
    from parser_test_function import is_channels_first_supported

    def train_and_save(model, name):
        x_train = np.random.rand(32, *model.input_shape[1:])
        y_train = np.random.rand(32, *model.output_shape[1:])
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
        model.fit(x_train, y_train, epochs=1, verbose=0)
        model.save(f"{dst_dir}/Sequential_{name}_test.keras")

    for act in ["relu", "elu", "leaky_relu", "selu", "sigmoid", "softmax", "swish", "tanh"]:
        model = models.Sequential([layers.Input(shape=(10,)), layers.Activation(act)])
        train_and_save(model, f"Activation_layer_{act.capitalize()}")

    if is_channels_first_supported():
        model = models.Sequential(
            [layers.Input(shape=(3, 8, 8)), layers.AveragePooling2D(pool_size=(2, 2), data_format="channels_first")]
        )
        train_and_save(model, "AveragePooling2D_channels_first")

    model = models.Sequential(
        [layers.Input(shape=(8, 8, 3)), layers.AveragePooling2D(pool_size=(2, 2), data_format="channels_last")]
    )
    train_and_save(model, "AveragePooling2D_channels_last")

    model = models.Sequential([layers.Input(shape=(10, 3, 5)), layers.BatchNormalization(axis=2)])
    train_and_save(model, "BatchNorm")

    if is_channels_first_supported():
        model = models.Sequential([layers.Input(shape=(3, 8, 8)), layers.Conv2D(4, (3, 3), data_format="channels_first")])
        train_and_save(model, "Conv2D_channels_first")

    model = models.Sequential(
        [layers.Input(shape=(8, 8, 3)), layers.Conv2D(4, (3, 3), data_format="channels_last", activation="tanh")]
    )
    train_and_save(model, "Conv2D_channels_last")

    model = models.Sequential(
        [layers.Input(shape=(8, 8, 3)), layers.Conv2D(4, (3, 3), padding="same", data_format="channels_last")]
    )
    train_and_save(model, "Conv2D_padding_same")

    model = models.Sequential(
        [layers.Input(shape=(8, 8, 3)), layers.Conv2D(4, (3, 3), padding="valid", data_format="channels_last")]
    )
    train_and_save(model, "Conv2D_padding_valid")

    model = models.Sequential([layers.Input(shape=(10,)), layers.Dense(5, activation="sigmoid")])
    train_and_save(model, "Dense")
    model = models.Sequential([layers.Input(shape=(10,)), layers.ELU(alpha=0.5)])
    train_and_save(model, "ELU")

    model = models.Sequential([layers.Input(shape=(4, 5)), layers.Flatten()])
    train_and_save(model, "Flatten")

    if is_channels_first_supported():
        model = models.Sequential([layers.Input(shape=(3, 4, 6)), layers.GlobalAveragePooling2D(data_format="channels_first")])
        train_and_save(model, "GlobalAveragePooling2D_channels_first")

    model = models.Sequential([layers.Input(shape=(4, 6, 3)), layers.GlobalAveragePooling2D(data_format="channels_last")])
    train_and_save(model, "GlobalAveragePooling2D_channels_last")

    model = models.Sequential([layers.Input(shape=(10,)), layers.LeakyReLU()])
    train_and_save(model, "LeakyReLU")

    if is_channels_first_supported():
        model = models.Sequential([layers.Input(shape=(3, 8, 8)), layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_first")])
        train_and_save(model, "MaxPool2D_channels_first")

    model = models.Sequential([layers.Input(shape=(8, 8, 3)), layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_last")])
    train_and_save(model, "MaxPool2D_channels_last")

    model = models.Sequential([layers.Input(shape=(10,)), layers.ReLU()])
    train_and_save(model, "ReLU")

    model = models.Sequential([layers.Input(shape=(4, 5)), layers.Reshape((2, 10))])
    train_and_save(model, "Reshape")

    model = models.Sequential([layers.Input(shape=(10,)), layers.Softmax()])
    train_and_save(model, "Softmax")

    model = models.Sequential(
        [
            layers.Input(shape=(32, 32, 3)),
            layers.Conv2D(16, (3, 3), padding="same", activation="swish"),
            layers.AveragePooling2D((2, 2), data_format="channels_last"),
            layers.GlobalAveragePooling2D(data_format="channels_last"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    train_and_save(model, "Layer_Combination_1")

    model = models.Sequential(
        [
            layers.Input(shape=(32, 32, 3)),
            layers.Conv2D(8, (3, 3), padding="valid", data_format="channels_last", activation="relu"),
            layers.MaxPooling2D((2, 2), data_format="channels_last"),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Reshape((16, 8)),
            layers.Flatten(),
            layers.Dense(32),
            layers.LeakyReLU(alpha=0.1),
            layers.Dense(10, activation="softmax"),
        ]
    )
    train_and_save(model, "Layer_Combination_2")

    model = models.Sequential(
        [
            layers.Input(shape=(8, 8, 1)),
            layers.Conv2D(4, (3, 3), padding="same", activation="relu", data_format="channels_last"),
            layers.AveragePooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(32, activation="elu"),
            layers.Dense(8, activation="swish"),
            layers.Dense(3, activation="softmax"),
        ]
    )
    train_and_save(model, "Layer_Combination_3")
