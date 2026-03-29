def generate_hls4ml_sequential(dst_dir):

    import numpy as np
    from keras import layers, models

    def train_and_save(model, name):
        x_train = np.random.rand(32, *model.input_shape[1:])
        y_train = np.random.rand(32, *model.output_shape[1:])
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
        model.fit(x_train, y_train, epochs=1, verbose=0)
        model.save(f"{dst_dir}/Sequential_{name}_test.keras")

    model = models.Sequential([layers.Input(shape=(10,)), layers.Dense(5)])
    train_and_save(model, "Dense")
