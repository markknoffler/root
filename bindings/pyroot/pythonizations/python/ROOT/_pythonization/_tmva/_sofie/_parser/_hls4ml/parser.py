import copy
import time
import math
import numpy as np


try:
    from .config import extract_hls_config, _activation_type_from_string
    from .schema import sofie_layer_dict
    from .layers.relu import MakeHLSReLU
    from .layers.elu import MakeHLSELU
    from .layers.gemm import MakeHLSGemm
    from .layers.permute import MakeHLSPermute
    from .layers.reshape import MakeHLSReshape
    from .layers.concat import MakeHLSConcat
    from .layers.batchnorm import MakeHLSBatchNorm
    from .layers.conv import MakeHLSConv
    from .layers.pooling import MakeHLSPooling
    from .layers.binary import MakeHLSBinary
    from .layers.sigmoid import MakeHLSSigmoid
    from .layers.tanh import MakeHLSTanh
    from .layers.softmax import MakeHLSSoftmax
    from .layers.swish import MakeHLSSwish
    from .layers.leaky_relu import MakeHLSLeakyRelu
    from .layers.selu import MakeHLSSeLU
except (ImportError, ValueError):
    from config import extract_hls_config, _activation_type_from_string
    from schema import sofie_layer_dict
    from layers.relu import MakeHLSReLU
    from layers.elu import MakeHLSELU
    from layers.gemm import MakeHLSGemm
    from layers.permute import MakeHLSPermute
    from layers.reshape import MakeHLSReshape
    from layers.concat import MakeHLSConcat
    from layers.batchnorm import MakeHLSBatchNorm
    from layers.conv import MakeHLSConv
    from layers.pooling import MakeHLSPooling
    from layers.binary import MakeHLSBinary
    from layers.sigmoid import MakeHLSSigmoid
    from layers.tanh import MakeHLSTanh
    from layers.softmax import MakeHLSSoftmax
    from layers.swish import MakeHLSSwish
    from layers.leaky_relu import MakeHLSLeakyRelu
    from layers.selu import MakeHLSSeLU


def MakeHLSActivation(layer):
    from ROOT.TMVA.Experimental import SOFIE
    attributes = layer["layerAttributes"]
    raw = attributes.get("Activation", attributes.get("activation", ""))
    fLayerActivation = _activation_type_from_string(raw)
    if fLayerActivation is None:
        fLayerActivation = str(raw).strip()
    if fLayerActivation == "ReLU":
        return MakeHLSReLU(layer)
    if fLayerActivation == "ELU":
        return MakeHLSELU(layer)
    if fLayerActivation == "SeLU":
        return MakeHLSSeLU(layer)
    if fLayerActivation == "Sigmoid":
        return MakeHLSSigmoid(layer)
    if fLayerActivation == "Tanh":
        return MakeHLSTanh(layer)
    if fLayerActivation == "Softmax":
        return MakeHLSSoftmax(layer)
    if fLayerActivation in ("Swish", "SiLU"):
        return MakeHLSSwish(layer)
    if fLayerActivation == "LeakyReLU":
        return MakeHLSLeakyRelu(layer)
    if str(fLayerActivation).lower() == "thresholdedrelu":
        # Fallback to ReLU if ThresholdedReLU is not yet supported in SOFIE
        return MakeHLSReLU(layer)
    else:
        raise Exception(
            "TMVA.SOFIE - HLS4ML activation " + fLayerActivation + " is not supported"
        )


mapHLS4MLLayer = {
    "ReLU": MakeHLSReLU,
    "relu": MakeHLSReLU,
    "ELU": MakeHLSELU,
    "elu": MakeHLSELU,
    "SeLU": MakeHLSSeLU,
    "selu": MakeHLSSeLU,
    "Sigmoid": MakeHLSSigmoid,
    "sigmoid": MakeHLSSigmoid,
    "Tanh": MakeHLSTanh,
    "tanh": MakeHLSTanh,
    "leaky_relu": MakeHLSLeakyRelu,
    "Softmax": MakeHLSSoftmax,
    "softmax": MakeHLSSoftmax,
    "Swish": MakeHLSSwish,
    "swish": MakeHLSSwish,
    "Dense": MakeHLSGemm,
    "BatchNormalization": MakeHLSBatchNorm,
    "batchnormalization": MakeHLSBatchNorm,
    "Conv2D": MakeHLSConv,
    "Conv1D": MakeHLSConv,
    "MaxPooling2D": MakeHLSPooling,
    "AveragePooling2D": MakeHLSPooling,
    "GlobalAveragePooling2D": MakeHLSPooling,
    "Permute": MakeHLSPermute,
    "Reshape": MakeHLSReshape,
    "Flatten": MakeHLSReshape,
    "Concatenate": MakeHLSConcat,
    "Concat": MakeHLSConcat,
    "Add": MakeHLSBinary,
    "Subtract": MakeHLSBinary,
    "Multiply": MakeHLSBinary,
}


def _move_op(op):
    import ROOT
    ROOT.SetOwnership(op, False)
    return ROOT.std.unique_ptr[type(op)](op)


def _add_blas_for_layer(rmodel, f_layer_type):
    if f_layer_type == "Dense":
        rmodel.AddBlasRoutines({"Gemm", "Gemv"})
    elif f_layer_type == "BatchNormalization":
        rmodel.AddBlasRoutines({"Copy", "Axpy"})
    elif f_layer_type in ("Conv1D", "Conv2D"):
        rmodel.AddBlasRoutines({"Gemm", "Axpy"})


def _register_initialised_tensors(rmodel, layer):
    # add initialised tensors from extracted cfg
    for tname, arr in (layer.get("initialisers") or {}).items():
        a = np.asarray(arr, dtype=np.float32)
        shape = [1] if a.ndim == 0 else list(a.shape)
        flat = np.ascontiguousarray(a.flatten(), dtype=np.float32)
        rmodel.AddInitializedTensor["float"](tname, shape, flat)


def add_layer_into_RModel(rmodel, layer_data, node_shapes):
    # add one layer into RModel
    layer_data = copy.deepcopy(layer_data)
    f_layer_type = layer_data["layerType"]
    attrs = layer_data["layerAttributes"]
    layer_name = attrs.get("name", "layer")

    if f_layer_type in ("Reshape", "Flatten"):
        target = attrs.get("target_shape")
        if target is None:
            target = [-1]
        target = np.ascontiguousarray(np.asarray(target, dtype="int64"))
        if f_layer_type == "Flatten":
            target = np.ascontiguousarray(np.asarray([-1], dtype="int64"))
        shape_name = layer_name + "_shape"
        rmodel.AddInitializedTensor["int64_t"](shape_name, [len(target)], target)

    if f_layer_type not in mapHLS4MLLayer and f_layer_type != "Activation":
        return rmodel

    inputs = list(layer_data["layerInput"])
    outputs = list(layer_data["layerOutput"])
    f_layer_output = outputs[0]
    channels_last = layer_data.get("channels_last", True)

    # Use node_shapes dictionary instead of rmodel.GetTensorShape
    input_name = inputs[0]
    if input_name in node_shapes:
        input_shape = list(node_shapes[input_name])
    else:
        # Fallback if not found, though it should be there
        try:
            input_shape = list(rmodel.GetTensorShape(input_name))
        except Exception:
            input_shape = [1, 1, 1, 1] 

    if f_layer_type == "GlobalAveragePooling2D":
        from ROOT.TMVA.Experimental import SOFIE
        if channels_last:
            rank = len(input_shape)
            axis = attrs.get("axis", -1)
            if axis < 0:
                axis += rank
            # Move channel axis to position 1
            perm = list(range(rank))
            perm.pop(axis)
            perm.insert(1, axis)
            
            # Register intermediate tensor
            pre_trans_name = layer_name + "PreTrans"
            new_shape = [input_shape[p] for p in perm]
            rmodel.AddIntermediateTensor(pre_trans_name, SOFIE.ETensorType.FLOAT, new_shape)
            node_shapes[pre_trans_name] = new_shape
            
            op = SOFIE.ROperator_Transpose("float")(perm, inputs[0], pre_trans_name)
            rmodel.AddOperator(_move_op(op))
            inputs[0] = pre_trans_name
            # Update input_shape for subsequent steps in this layer
            input_shape = new_shape
            
        outputs[0] = layer_name + "Squeeze"
        layer_data["layerInput"] = inputs
        layer_data["layerOutput"] = outputs
        
        # Add Pooling operator
        pool_op = mapHLS4MLLayer[f_layer_type](layer_data)
        pool_out_shape = input_shape[:2] + [1, 1]
        rmodel.AddIntermediateTensor(outputs[0], SOFIE.ETensorType.FLOAT, pool_out_shape)
        node_shapes[outputs[0]] = pool_out_shape
        rmodel.AddOperator(_move_op(pool_op))
        
        # Squeeze to final output
        op = SOFIE.ROperator_Reshape(SOFIE.ReshapeOpMode.Squeeze, [2, 3], outputs[0], f_layer_output)
        # Register final output tensor
        final_output_shape = [input_shape[0], input_shape[1]]
        rmodel.AddIntermediateTensor(f_layer_output, SOFIE.ETensorType.FLOAT, final_output_shape)
        node_shapes[f_layer_output] = final_output_shape
        rmodel.AddOperator(_move_op(op))
        return rmodel

    if f_layer_type == "BatchNormalization":
        from ROOT.TMVA.Experimental import SOFIE
        rank = len(input_shape)
        axis = attrs.get("axis", -1)
        if isinstance(axis, (list, tuple)):
            axis = axis[0]
        axis = int(axis)
        if axis < 0:
            axis += rank
            
        # Move normalization axis to position 1 if it isn't already
        perm = list(range(0, rank))
        if axis != 1 and axis < rank:
            perm.pop(axis)
            perm.insert(1, axis)
            
            pre_trans_name = layer_name + "PreTrans"
            new_shape = [input_shape[p] for p in perm]
            rmodel.AddIntermediateTensor(pre_trans_name, SOFIE.ETensorType.FLOAT, new_shape)
            node_shapes[pre_trans_name] = new_shape
            
            op = SOFIE.ROperator_Transpose("float")(perm, inputs[0], pre_trans_name)
            rmodel.AddOperator(_move_op(op))
            inputs[0] = pre_trans_name
            outputs[0] = layer_name + "PostTrans"
            rmodel.AddIntermediateTensor(outputs[0], SOFIE.ETensorType.FLOAT, new_shape)
            node_shapes[outputs[0]] = new_shape
            
            layer_data["layerInput"] = inputs
            layer_data["layerOutput"] = outputs
            rmodel.AddOperator(_move_op(mapHLS4MLLayer[f_layer_type](layer_data)))
            
            # Transpose back
            inv_perm = [0] * rank
            for i, p in enumerate(perm):
                inv_perm[p] = i
            rmodel.AddIntermediateTensor(f_layer_output, SOFIE.ETensorType.FLOAT, input_shape)
            node_shapes[f_layer_output] = input_shape
            op = SOFIE.ROperator_Transpose("float")(inv_perm, outputs[0], f_layer_output)
            rmodel.AddOperator(_move_op(op))
        else:
            rmodel.AddIntermediateTensor(f_layer_output, SOFIE.ETensorType.FLOAT, input_shape)
            node_shapes[f_layer_output] = input_shape
            rmodel.AddOperator(_move_op(mapHLS4MLLayer[f_layer_type](layer_data)))
        return rmodel

    if f_layer_type in ("MaxPooling2D", "AveragePooling2D"):
        from ROOT.TMVA.Experimental import SOFIE
        rank = len(input_shape)
        if channels_last:
            # Default to axis -1 for channels_last
            axis = rank - 1
            perm = list(range(rank))
            perm.pop(axis)
            perm.insert(1, axis)
            
            pre_trans_name = layer_name + "PreTrans"
            new_shape = [input_shape[p] for p in perm]
            rmodel.AddIntermediateTensor(pre_trans_name, SOFIE.ETensorType.FLOAT, new_shape)
            node_shapes[pre_trans_name] = new_shape
            
            op = SOFIE.ROperator_Transpose("float")(perm, inputs[0], pre_trans_name)
            rmodel.AddOperator(_move_op(op))
            inputs[0] = pre_trans_name
            outputs[0] = layer_name + "PostTrans"
            
            layer_data["layerInput"] = inputs
            layer_data["layerOutput"] = outputs
            pool_op = mapHLS4MLLayer[f_layer_type](layer_data)
            
            k = attrs.get("pool_size", [2, 2])
            s = attrs.get("strides", k)
            p = attrs.get("padding", "valid")
            
            if p == "valid":
                h_out = (new_shape[2] - k[0]) // s[0] + 1
                w_out = (new_shape[3] - k[1]) // s[1] + 1
            else:
                h_out = math.ceil(new_shape[2] / s[0])
                w_out = math.ceil(new_shape[3] / s[1])
            
            pool_out_shape = [new_shape[0], new_shape[1], h_out, w_out]
            rmodel.AddIntermediateTensor(outputs[0], SOFIE.ETensorType.FLOAT, pool_out_shape)
            node_shapes[outputs[0]] = pool_out_shape
            rmodel.AddOperator(_move_op(pool_op))
            
            inv_perm = [0] * rank
            for i, p_val in enumerate(perm):
                inv_perm[p_val] = i
            
            # Map back to original layout
            final_out_shape = [pool_out_shape[inv_p] for inv_p in inv_perm]
            rmodel.AddIntermediateTensor(f_layer_output, SOFIE.ETensorType.FLOAT, final_out_shape)
            node_shapes[f_layer_output] = final_out_shape
            op = SOFIE.ROperator_Transpose("float")(inv_perm, outputs[0], f_layer_output)
            rmodel.AddOperator(_move_op(op))
        else:
            # For channels_first, compute output shape and register
            k = attrs.get("pool_size", [2, 2])
            s = attrs.get("strides", k)
            p = attrs.get("padding", "valid")
            if p == "valid":
                h_out = (input_shape[2] - k[0]) // s[0] + 1
                w_out = (input_shape[3] - k[1]) // s[1] + 1
            else:
                h_out = math.ceil(input_shape[2] / s[0])
                w_out = math.ceil(input_shape[3] / s[1])
            output_shape = [input_shape[0], input_shape[1], h_out, w_out]
            rmodel.AddIntermediateTensor(f_layer_output, SOFIE.ETensorType.FLOAT, output_shape)
            node_shapes[f_layer_output] = output_shape
            rmodel.AddOperator(_move_op(mapHLS4MLLayer[f_layer_type](layer_data)))
        return rmodel

    if f_layer_type in ("Conv1D", "Conv2D"):
        from ROOT.TMVA.Experimental import SOFIE
        rank = len(input_shape)
        if channels_last:
            axis = rank - 1
            perm = list(range(rank))
            perm.pop(axis)
            perm.insert(1, axis)
                
            pre_trans_name = layer_name + "PreTrans"
            new_shape = [input_shape[p] for p in perm]
            rmodel.AddIntermediateTensor(pre_trans_name, SOFIE.ETensorType.FLOAT, new_shape)
            node_shapes[pre_trans_name] = new_shape
            
            op = SOFIE.ROperator_Transpose("float")(perm, inputs[0], pre_trans_name)
            rmodel.AddOperator(_move_op(op))
            inputs[0] = pre_trans_name
            layer_data["layerInput"] = inputs
            input_shape = new_shape
        
        if channels_last:
            post_name = layer_name + "PostTrans"
        else:
            post_name = f_layer_output
            
        outputs[0] = post_name
        layer_data["layerOutput"] = outputs
        
        conv_op = mapHLS4MLLayer[f_layer_type](layer_data)
        m_filters = attrs.get("n_filt", 1)
        k = attrs.get("kernel_size", [3, 3])
        if isinstance(k, int): k = [k, k]
        s = attrs.get("strides", [1, 1])
        if isinstance(s, int): s = [s, s]
        pad = attrs.get("padding", "valid")
        
        if rank == 4:
            h_in, w_in = input_shape[2], input_shape[3]
            if pad == "valid":
                h_out = (h_in - k[0]) // s[0] + 1
                w_out = (w_in - k[1]) // s[1] + 1
            else:
                h_out = math.ceil(h_in / s[0])
                w_out = math.ceil(w_in / s[1])
            conv_out_shape = [input_shape[0], m_filters, h_out, w_out]
        else: # Conv1D or other
            l_in = input_shape[-1] # Fallback if rank is not 3
            if rank >= 3:
                l_in = input_shape[2]
                
            if pad == "valid":
                l_out = (l_in - k[0]) // s[0] + 1
            else:
                l_out = math.ceil(l_in / s[0])
            
            if rank == 3:
                conv_out_shape = [input_shape[0], m_filters, l_out]
            else:
                # Fallback for unexpected rank
                conv_out_shape = [input_shape[0], m_filters] + [l_out] * (rank - 2)

        rmodel.AddIntermediateTensor(post_name, SOFIE.ETensorType.FLOAT, conv_out_shape)
        node_shapes[post_name] = conv_out_shape
        rmodel.AddOperator(_move_op(conv_op))
        
        if channels_last:
            inv_perm = [0] * rank
            for i, p_val in enumerate(perm):
                inv_perm[p_val] = i
            
            final_out_shape = [conv_out_shape[inv_p] for inv_p in inv_perm]
            rmodel.AddIntermediateTensor(f_layer_output, SOFIE.ETensorType.FLOAT, final_out_shape)
            node_shapes[f_layer_output] = final_out_shape
            op = SOFIE.ROperator_Transpose("float")(inv_perm, post_name, f_layer_output)
            rmodel.AddOperator(_move_op(op))
        return rmodel

    if f_layer_type == "Activation":
        from ROOT.TMVA.Experimental import SOFIE
        op = MakeHLSActivation(layer_data)
        rmodel.AddIntermediateTensor(f_layer_output, SOFIE.ETensorType.FLOAT, input_shape)
        node_shapes[f_layer_output] = input_shape
        rmodel.AddOperator(_move_op(op))
        return rmodel

    # Universal registration for remaining layers
    from ROOT.TMVA.Experimental import SOFIE
    op = mapHLS4MLLayer[f_layer_type](layer_data)
    
    try:
        if f_layer_type == "Dense":
            # For Dense, it's [N, out_dim]
            # Use hls4ml metadata directly for more accuracy
            out_dim = attrs.get("n_out", 1)
            output_shape = [input_shape[0], out_dim]
        elif f_layer_type == "Reshape":
            target = attrs.get("target_shape", [1])
            output_shape = [input_shape[0]] + list(target)
        elif f_layer_type == "Flatten":
            prod = 1
            for x in input_shape[1:]: prod *= x
            output_shape = [input_shape[0], prod]
        elif f_layer_type == "Concatenate":
            output_shape = list(input_shape)
            axis = attrs.get("axis", -1)
            if axis < 0: axis += len(input_shape)
            concat_dim = 0
            for inp_name in inputs:
                concat_dim += node_shapes.get(inp_name, [0,0,0,0])[axis]
            output_shape[axis] = concat_dim
        else:
            output_shape = input_shape
            
        rmodel.AddIntermediateTensor(f_layer_output, SOFIE.ETensorType.FLOAT, output_shape)
        node_shapes[f_layer_output] = output_shape
    except Exception:
        pass

    rmodel.AddOperator(_move_op(op))
    return rmodel


def build_rmodel(cfg, name=None):
    import ROOT
    from ROOT.TMVA.Experimental import SOFIE
    # build RModel using cfg only
    model_name = name or cfg.get("name", "HLS4MLModel")
    parsetime = time.asctime(time.gmtime(time.time()))
    rmodel = SOFIE.RModel.RModel(model_name, parsetime)

    node_shapes = {}

    # Register all inputs properly
    input_names = cfg.get("inputs", [])
    if not input_names:
        input_names = ["input_0"]
    
    for inp_name in input_names:
        # Get shape from cfg, default to [1, 1] if missing
        raw_shape = cfg.get("input_node_shapes", {}).get(inp_name)
        if raw_shape is None:
            raw_shape = cfg.get("input_shape", [1, 1])
        
        # Ensure we have a batch dimension of 1 if rank is too low
        # Keras usually provides [H, W, C] but SOFIE wants [B, C, H, W]
        # or [B, H, W, C] depending on the layer initialization.
        # We enforce B=1 at index 0 if not present.
        sanitized_shape = []
        for x in raw_shape:
            try:
                val = int(x)
                if val is None or val <= 0: val = 1
                sanitized_shape.append(val)
            except Exception:
                sanitized_shape.append(1)
        
        # If rank is 1, treat as [1, N]
        # If rank is 3, treat as [1, H, W, C]
        if len(sanitized_shape) == 1:
            sanitized_shape = [1, sanitized_shape[0]]
        elif len(sanitized_shape) == 3:
             sanitized_shape = [1] + sanitized_shape
        
        rmodel.AddInputTensorInfo(inp_name, SOFIE.ETensorType.FLOAT, sanitized_shape)
        rmodel.AddInputTensorName(inp_name)
        node_shapes[inp_name] = sanitized_shape
        print(f"DEBUG: Registered input {inp_name} with shape {sanitized_shape}")

    rmodel.AddBlasRoutines({"Gemm", "Gemv"})

    for layer in cfg.get("layers", []):
        if layer.get("layerType") == "Input":
            continue
        lt = layer.get("layerType")
        _add_blas_for_layer(rmodel, lt)
        if lt in ("SeLU", "selu", "Sigmoid", "sigmoid"):
            rmodel.AddNeededStdLib("cmath")
        _register_initialised_tensors(rmodel, layer)
        rmodel = add_layer_into_RModel(rmodel, sofie_layer_dict(layer), node_shapes)

    output_names = list(cfg.get("outputs", []))
    if not output_names:
        layers = cfg.get("layers", [])
        for L in reversed(layers):
            outs = L.get("layerOutput", [])
            if outs:
                output_names = list(outs)
                break
    if output_names:
        rmodel.AddOutputTensorNameList(output_names)

    # Initialize model to register all intermediate tensors
    try:
        rmodel.Initialize()
    except Exception:
        try:
            rmodel.Initialize(1)
        except Exception:
            pass

    return rmodel


class PyHLS4ML:
    @staticmethod
    def ParseFromModelGraph(hls_model, name=None, keras_model=None):
        cfg = extract_hls_config(hls_model, keras_model=keras_model)
        return build_rmodel(cfg, name=name)

    @staticmethod
    def ExtractConfig(hls_model, keras_model=None):
        # extract canonical cfg
        return extract_hls_config(hls_model, keras_model=keras_model)

    @staticmethod
    def BuildFromConfig(cfg, name=None):
        # build from extracted cfg
        return build_rmodel(cfg, name=name)
