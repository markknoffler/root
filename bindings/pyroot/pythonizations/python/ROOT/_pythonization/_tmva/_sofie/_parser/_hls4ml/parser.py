import copy
import time
import numpy as np
import math


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


def add_layer_into_RModel(rmodel, layer_data):
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

    if f_layer_type == "GlobalAveragePooling2D":
        from ROOT.TMVA.Experimental import SOFIE
        if channels_last:
            input_shape = list(rmodel.GetTensorShape(inputs[0]))
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
            
            op = SOFIE.ROperator_Transpose("float")(perm, inputs[0], pre_trans_name)
            rmodel.AddOperator(_move_op(op))
            inputs[0] = pre_trans_name
            
        outputs[0] = layer_name + "Squeeze"
        layer_data["layerInput"] = inputs
        layer_data["layerOutput"] = outputs
        
        # Add Pooling operator
        pool_op = mapHLS4MLLayer[f_layer_type](layer_data)
        rmodel.AddIntermediateTensor(outputs[0], SOFIE.ETensorType.FLOAT, list(rmodel.GetTensorShape(inputs[0])[:2]) + [1, 1])
        rmodel.AddOperator(_move_op(pool_op))
        
        # Squeeze to final output
        op = SOFIE.ROperator_Reshape(SOFIE.ReshapeOpMode.Squeeze, [2, 3], outputs[0], f_layer_output)
        # Register final output tensor
        rmodel.AddIntermediateTensor(f_layer_output, SOFIE.ETensorType.FLOAT, [input_shape[0], input_shape[1]])
        rmodel.AddOperator(_move_op(op))
        return rmodel

    if f_layer_type == "BatchNormalization":
        from ROOT.TMVA.Experimental import SOFIE
        input_shape = list(rmodel.GetTensorShape(inputs[0]))
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
            
            op = SOFIE.ROperator_Transpose("float")(perm, inputs[0], pre_trans_name)
            rmodel.AddOperator(_move_op(op))
            inputs[0] = pre_trans_name
            outputs[0] = layer_name + "PostTrans"
            rmodel.AddIntermediateTensor(f_layer_output, SOFIE.ETensorType.FLOAT, input_shape)
            rmodel.AddIntermediateTensor(outputs[0], SOFIE.ETensorType.FLOAT, new_shape)
            
            layer_data["layerInput"] = inputs
            layer_data["layerOutput"] = outputs
            rmodel.AddOperator(_move_op(mapHLS4MLLayer[f_layer_type](layer_data)))
            
            # Transpose back
            inv_perm = [0] * rank
            for i, p in enumerate(perm):
                inv_perm[p] = i
            rmodel.AddIntermediateTensor(f_layer_output, SOFIE.ETensorType.FLOAT, input_shape)
            op = SOFIE.ROperator_Transpose("float")(inv_perm, outputs[0], f_layer_output)
            rmodel.AddOperator(_move_op(op))
        else:
            rmodel.AddOperator(_move_op(mapHLS4MLLayer[f_layer_type](layer_data)))
        return rmodel

    if f_layer_type in ("MaxPooling2D", "AveragePooling2D"):
        from ROOT.TMVA.Experimental import SOFIE
        input_shape = list(rmodel.GetTensorShape(inputs[0]))
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
            
            op = SOFIE.ROperator_Transpose("float")(perm, inputs[0], pre_trans_name)
            rmodel.AddOperator(_move_op(op))
            inputs[0] = pre_trans_name
            outputs[0] = layer_name + "PostTrans"
            
            layer_data["layerInput"] = inputs
            layer_data["layerOutput"] = outputs
            pool_op = mapHLS4MLLayer[f_layer_type](layer_data)
            
            # We need the output shape of pooling to register PostTrans
            # For pooling, it's roughly [N, C, H_out, W_out]
            # Since we don't have an easy way to compute H_out/W_out here without repeating logic,
            # we'll use a placeholder and rely on Initialize() if possible, 
            # but SOFIE usually wants the shape upfront.
            # Let's try to infer from attributes.
            k = attrs.get("pool_size", [2, 2])
            s = attrs.get("strides", k)
            p = attrs.get("padding", "valid")
            h_out = input_shape[1] # Placeholder
            w_out = input_shape[2] # Placeholder
            # (Simplified pool shape calculation)
            if p == "valid":
                h_out = (input_shape[perm[2]] - k[0]) // s[0] + 1
                w_out = (input_shape[perm[3]] - k[1]) // s[1] + 1
            else:
                h_out = math.ceil(input_shape[perm[2]] / s[0])
                w_out = math.ceil(input_shape[perm[3]] / s[1])
            
            pool_out_shape = [input_shape[0], input_shape[axis], h_out, w_out]
            rmodel.AddIntermediateTensor(outputs[0], SOFIE.ETensorType.FLOAT, pool_out_shape)
            rmodel.AddOperator(_move_op(pool_op))
            
            inv_perm = [0] * rank
            for i, p_val in enumerate(perm):
                inv_perm[p_val] = i
            
            # Map back to original layout
            final_out_shape = [pool_out_shape[inv_p] for inv_p in inv_perm]
            rmodel.AddIntermediateTensor(f_layer_output, SOFIE.ETensorType.FLOAT, final_out_shape)
            op = SOFIE.ROperator_Transpose("float")(inv_perm, outputs[0], f_layer_output)
            rmodel.AddOperator(_move_op(op))
        else:
            rmodel.AddOperator(_move_op(mapHLS4MLLayer[f_layer_type](layer_data)))
        return rmodel

    if f_layer_type in ("Conv1D", "Conv2D"):
        from ROOT.TMVA.Experimental import SOFIE
        input_shape = list(rmodel.GetTensorShape(inputs[0]))
        rank = len(input_shape)
        if channels_last:
            axis = rank - 1
            
            perm = list(range(rank))
            perm.pop(axis)
            perm.insert(1, axis)
                
            pre_trans_name = layer_name + "PreTrans"
            new_shape = [input_shape[p] for p in perm]
            rmodel.AddIntermediateTensor(pre_trans_name, SOFIE.ETensorType.FLOAT, new_shape)
            
            op = SOFIE.ROperator_Transpose("float")(perm, inputs[0], pre_trans_name)
            rmodel.AddOperator(_move_op(op))
            inputs[0] = pre_trans_name
            layer_data["layerInput"] = inputs
        
        if channels_last:
            post_name = layer_name + "PostTrans"
        else:
            post_name = f_layer_output
            
        outputs[0] = post_name
        layer_data["layerOutput"] = outputs
        
        conv_op = mapHLS4MLLayer[f_layer_type](layer_data)
        # Register Conv output tensor. Shape: [N, M, H_out, W_out]
        # M is number of filters.
        m_filters = attrs.get("n_filt", 1)
        # Calculate H_out, W_out
        k = attrs.get("kernel_size", [3, 3])
        s = attrs.get("strides", [1, 1])
        pad = attrs.get("padding", "valid")
        if rank == 4:
            h_in, w_in = input_shape[1], input_shape[2]
            if pad == "valid":
                h_out = (h_in - k[0]) // s[0] + 1
                w_out = (w_in - k[1]) // s[1] + 1
            else:
                h_out = math.ceil(h_in / s[0])
                w_out = math.ceil(w_in / s[1])
            conv_out_shape = [input_shape[0], m_filters, h_out, w_out]
        else: # Conv1D
            l_in = input_shape[1]
            if pad == "valid":
                l_out = (l_in - k[0]) // s[0] + 1
            else:
                l_out = math.ceil(l_in / s[0])
            conv_out_shape = [input_shape[0], m_filters, l_out]

        rmodel.AddIntermediateTensor(post_name, SOFIE.ETensorType.FLOAT, conv_out_shape)
        rmodel.AddOperator(_move_op(conv_op))
        
        if channels_last:
            inv_perm = [0] * rank
            for i, p_val in enumerate(perm):
                inv_perm[p_val] = i
            
            final_out_shape = [conv_out_shape[inv_p] for inv_p in inv_perm]
            rmodel.AddIntermediateTensor(f_layer_output, SOFIE.ETensorType.FLOAT, final_out_shape)
            op = SOFIE.ROperator_Transpose("float")(inv_perm, post_name, f_layer_output)
            rmodel.AddOperator(_move_op(op))
        return rmodel

    if f_layer_type == "Activation":
        op = MakeHLSActivation(layer_data)
        # Register output tensor for standalone Activation layers
        input_shape = list(rmodel.GetTensorShape(inputs[0]))
        rmodel.AddIntermediateTensor(f_layer_output, SOFIE.ETensorType.FLOAT, input_shape)
        rmodel.AddOperator(_move_op(op))
        return rmodel

    op = mapHLS4MLLayer[f_layer_type](layer_data)
    # Register output tensor for other layers (Dense, Binary ops, etc.)
    if f_layer_type in ["Dense", "ReLU", "ELU", "LeakyReLU", "SeLU", "Sigmoid", "Tanh", "Softmax", "Swish", "Add", "Subtract", "Multiply"]:
        # For Dense, it's [N, out_dim]
        if f_layer_type == "Dense":
            w_shape = list(rmodel.GetTensorShape(layer_data["layerWeight"][0]))
            out_dim = w_shape[0] if attrs.get("transpose_weights", False) else w_shape[1]
            input_shape = list(rmodel.GetTensorShape(inputs[0]))
            rmodel.AddIntermediateTensor(f_layer_output, SOFIE.ETensorType.FLOAT, [input_shape[0], out_dim])
        else:
            # Most activations and binary ops preserve shape
            input_shape = list(rmodel.GetTensorShape(inputs[0]))
            rmodel.AddIntermediateTensor(f_layer_output, SOFIE.ETensorType.FLOAT, input_shape)

    rmodel.AddOperator(_move_op(op))
    return rmodel


def build_rmodel(cfg, name=None):
    import ROOT
    from ROOT.TMVA.Experimental import SOFIE
    # build RModel using cfg only
    model_name = name or cfg.get("name", "HLS4MLModel")
    parsetime = time.asctime(time.gmtime(time.time()))
    rmodel = SOFIE.RModel.RModel(model_name, parsetime)

    input_names = cfg.get("inputs", [])
    if not input_names:
        input_names = ["input_0"]
    input_shape = cfg.get("input_shape", [1])
    if len(input_shape) == 1:
        input_shape = [1, input_shape[0]]

    rmodel.AddInputTensorInfo(input_names[0], SOFIE.ETensorType.FLOAT, input_shape)
    rmodel.AddInputTensorName(input_names[0])
    rmodel.AddBlasRoutines({"Gemm", "Gemv"})

    for layer in cfg.get("layers", []):
        if layer.get("layerType") == "Input":
            continue
        lt = layer.get("layerType")
        _add_blas_for_layer(rmodel, lt)
        if lt in ("SeLU", "selu", "Sigmoid", "sigmoid"):
            rmodel.AddNeededStdLib("cmath")
        _register_initialised_tensors(rmodel, layer)
        rmodel = add_layer_into_RModel(rmodel, sofie_layer_dict(layer))

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
    rmodel.Initialize()

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
