import math

from .. import get_keras_version


# handles the Conv2DTranspose layer (sometimes called deconvolution)
# its basically the reverse of a regular conv, used a lot in upsampling / generative models
def MakeKerasConvTranspose(layer):
    from ROOT.TMVA.Experimental import SOFIE

    keras_version = get_keras_version()

    # pull out input/output names and dtype
    finput = layer["layerInput"]
    foutput = layer["layerOutput"]
    fLayerDType = layer["layerDType"]
    fLayerInputName = finput[0]
    fLayerOutputName = foutput[0]

    attributes = layer["layerAttributes"]

    # first weight is the kernel, second is the bias (bias might not exist)
    fWeightNames = layer["layerWeight"]
    fKernelName = fWeightNames[0]
    fBiasName = fWeightNames[1] if len(fWeightNames) > 1 else ""

    fAttrDilations = list(attributes["dilation_rate"])
    fAttrGroup = int(attributes["groups"])
    fAttrKernelShape = list(attributes["kernel_size"])
    fAttrStrides = list(attributes["strides"])
    fKerasPadding = str(attributes["padding"])

    # keras doesn't expose output_padding directly so just set it to zero
    fAttrOutputPadding = [0, 0]
    fAttrOutputShape = []

    if fKerasPadding == "valid":
        fAttrAutopad = "VALID"
        fAttrPads = []
    elif fKerasPadding == "same":
        fAttrAutopad = "NOTSET"

        # need to know the input spatial size to compute the padding
        if keras_version < "2.16":
            fInputShape = attributes["_build_input_shape"]
        else:
            fInputShape = attributes["_build_shapes_dict"]["input_shape"]

        # keras stores input as (batch, H, W, C) - channels last
        inputHeight = fInputShape[1]
        inputWidth = fInputShape[2]

        # for a transposed conv with same padding, output = input * stride
        # the padding needed is: kernel - stride (clamped to 0)
        padding_height = max(fAttrKernelShape[0] - fAttrStrides[0], 0)
        padding_width = max(fAttrKernelShape[1] - fAttrStrides[1], 0)

        pad_top = math.floor(padding_height / 2)
        pad_bottom = padding_height - pad_top
        pad_left = math.floor(padding_width / 2)
        pad_right = padding_width - pad_left

        fAttrPads = [pad_top, pad_left, pad_bottom, pad_right]
    else:
        raise RuntimeError(
            "TMVA::SOFIE - RModel Keras Parser doesn't yet support Conv2DTranspose with padding " + fKerasPadding
        )

    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        op = SOFIE.ROperator_ConvTranspose["float"](
            fAttrAutopad,
            fAttrDilations,
            fAttrGroup,
            fAttrKernelShape,
            fAttrOutputPadding,
            fAttrOutputShape,
            fAttrPads,
            fAttrStrides,
            fLayerInputName,
            fKernelName,
            fBiasName,
            fLayerOutputName,
        )
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator Conv2DTranspose does not yet support input type " + fLayerDType
        )
