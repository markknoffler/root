def generate_hls4ml_functional(dst_dir):

    import numpy as np
    from keras import layers, models

    def train_and_save(model, name):
        if isinstance(model.input_shape, list):
            x_train = [np.random.rand(32, *shape[1:]) for shape in model.input_shape]
        else:
            x_train = np.random.rand(32, *model.input_shape[1:])
        y_train = np.random.rand(32, *model.output_shape[1:])
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
        model.fit(x_train, y_train, epochs=1, verbose=0)
        model.save(f"{dst_dir}/Functional_{name}_test.keras")

    inp = layers.Input(shape=(10,))
    train_and_save(models.Model(inp, layers.Dense(5)(inp)), "Dense")
