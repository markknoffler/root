def extract_hls_config(hls_model):
    layers = []
    weights = {}

    name = getattr(hls_model, "name", None)
    if name is None:
        name = "hls_model"

    if hasattr(hls_model, "get_layers"):
        hls_layers = hls_model.get_layers()
    else:
        hls_layers = getattr(hls_model, "layers", [])

    for layer in hls_layers:
        layer_name = getattr(layer, "name", "")
        layer_type = type(layer).__name__

        attrs = {}
        if hasattr(layer, "attributes"):
            try:
                for k, v in layer.attributes.items():
                    key = str(k)
                    try:
                        attrs[key] = v
                    except Exception:
                        attrs[key] = str(v)
            except Exception:
                attrs = {}

        inputs = []
        if hasattr(layer, "inputs"):
            for x in layer.inputs:
                n = getattr(x, "name", None)
                if n is None:
                    n = str(x)
                inputs.append(n)

        outputs = []
        if hasattr(layer, "outputs"):
            for x in layer.outputs:
                n = getattr(x, "name", None)
                if n is None:
                    n = str(x)
                outputs.append(n)

        shape = None
        if hasattr(layer, "get_output_variable"):
            try:
                var = layer.get_output_variable()
                if hasattr(var, "shape"):
                    try:
                        shape = list(var.shape)
                    except Exception:
                        shape = None
            except Exception:
                shape = None

        layers.append(
            {
                "name": layer_name,
                "class_name": layer_type,
                "attributes": attrs,
                "inputs": inputs,
                "outputs": outputs,
                "output_shape": shape,
            }
        )

        if hasattr(layer, "get_weights"):
            try:
                wdict = layer.get_weights()
            except Exception:
                wdict = {}
            if isinstance(wdict, dict):
                for wname, wval in wdict.items():
                    key = layer_name + "/" + str(wname)
                    try:
                        s = list(wval.shape)
                    except Exception:
                        try:
                            s = list(wval)
                        except Exception:
                            s = []
                    weights[key] = {"shape": s}

    inputs = []
    if hasattr(hls_model, "inputs"):
        for x in hls_model.inputs:
            n = getattr(x, "name", None)
            if n is None:
                n = str(x)
            inputs.append(n)

    outputs = []
    if hasattr(hls_model, "outputs"):
        for x in hls_model.outputs:
            n = getattr(x, "name", None)
            if n is None:
                n = str(x)
            outputs.append(n)

    return {
        "name": name,
        "layers": layers,
        "weights": weights,
        "inputs": inputs,
        "outputs": outputs,
    }

