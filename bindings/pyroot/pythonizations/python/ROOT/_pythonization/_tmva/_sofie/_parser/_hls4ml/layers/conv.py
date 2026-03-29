import math
def MakeHLSConv(layer):
    from ROOT.TMVA.Experimental import SOFIE
    # build Conv op
    finput = layer["layerInput"]
    foutput = layer["layerOutput"]
    fLayerDType = layer["layerDType"]
    fLayerInputName = finput[0]
    fLayerOutputName = foutput[0]
    attributes = layer["layerAttributes"]
    fWeightNames = layer["layerWeight"]
    fKernelName = fWeightNames[0]
    fBiasName = fWeightNames[1]
    fAttrDilations = list(attributes["dilation_rate"])
    fAttrGroup = int(attributes["groups"])
    fAttrKernelShape = list(attributes["kernel_size"])
    fKerasPadding = str(attributes["padding"])
    fAttrStrides = list(attributes["strides"])
    fAttrPads = []

    if fKerasPadding == "valid":
        fAttrAutopad = "VALID"
    elif fKerasPadding == "same":
        fAttrAutopad = "NOTSET"
        fInputShape = attributes.get("_build_input_shape") or attributes.get("input_shape_for_padding")
        if not fInputShape:
            raise RuntimeError("TMVA::SOFIE - Conv 'same' padding requires input shape in layer attributes")
        if len(fAttrKernelShape) == 1:
            input_len = fInputShape[1]
            output_len = math.ceil(float(input_len) / float(fAttrStrides[0]))
            pad_total = max((output_len - 1) * fAttrStrides[0] + fAttrKernelShape[0] - input_len, 0)
            pad_before = math.floor(pad_total / 2)
            pad_after = pad_total - pad_before
            fAttrPads = [pad_before, pad_after]
        else:
            input_height = fInputShape[1]
            input_width = fInputShape[2]
            output_height = math.ceil(float(input_height) / float(fAttrStrides[0]))
            output_width = math.ceil(float(input_width) / float(fAttrStrides[1]))
            padding_height = max((output_height - 1) * fAttrStrides[0] + fAttrKernelShape[0] - input_height, 0)
            padding_width = max((output_width - 1) * fAttrStrides[1] + fAttrKernelShape[1] - input_width, 0)
            padding_top = math.floor(padding_height / 2)
            padding_bottom = padding_height - padding_top
            padding_left = math.floor(padding_width / 2)
            padding_right = padding_width - padding_left
            fAttrPads = [padding_top, padding_bottom, padding_left, padding_right]
    else:
        raise RuntimeError("TMVA::SOFIE - Conv padding " + fKerasPadding + " is not supported")

    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        return SOFIE.ROperator_Conv["float"](
            fAttrAutopad,
            fAttrDilations,
            fAttrGroup,
            fAttrKernelShape,
            fAttrPads,
            fAttrStrides,
            fLayerInputName,
            fKernelName,
            fBiasName,
            fLayerOutputName,
        )
    raise RuntimeError("TMVA::SOFIE - Operator Conv does not support input type " + str(fLayerDType))
