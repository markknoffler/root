def MakeHLSPooling(layer):
    from ROOT.TMVA.Experimental import SOFIE
    # build pooling op
    fLayerDType = layer["layerDType"]
    finput = layer["layerInput"]
    foutput = layer["layerOutput"]
    fLayerType = layer["layerType"]
    fLayerInputName = finput[0]
    fLayerOutputName = foutput[0]
    pool_atrr = SOFIE.RAttributes_Pool()
    attributes = layer["layerAttributes"]
    fAttrKernelShape = []
    fKerasPadding = "valid"
    fAttrStrides = []
    if fLayerType != "GlobalAveragePooling2D":
        fAttrKernelShape = list(attributes["pool_size"])
        fKerasPadding = str(attributes["padding"])
        fAttrStrides = list(attributes["strides"])

    fAttrDilations = (1, 1)
    fpads = [0, 0, 0, 0, 0, 0]
    pool_atrr.ceil_mode = 0
    pool_atrr.count_include_pad = 0
    pool_atrr.storage_order = 0

    if fKerasPadding == "valid":
        fAttrAutopad = "VALID"
    elif fKerasPadding == "same":
        fAttrAutopad = "NOTSET"
    else:
        raise RuntimeError("TMVA::SOFIE - Pooling padding " + fKerasPadding + " is not supported")

    pool_atrr.dilations = list(fAttrDilations)
    pool_atrr.strides = list(fAttrStrides)
    pool_atrr.pads = fpads
    pool_atrr.kernel_shape = list(fAttrKernelShape)
    pool_atrr.auto_pad = fAttrAutopad

    if "Max" in fLayerType and "Global" not in fLayerType:
        PoolMode = SOFIE.PoolOpMode.MaxPool
    elif "AveragePool" in fLayerType and "Global" not in fLayerType:
        PoolMode = SOFIE.PoolOpMode.AveragePool
    elif "GlobalAverage" in fLayerType:
        PoolMode = SOFIE.PoolOpMode.GloabalAveragePool
    else:
        raise RuntimeError("TMVA::SOFIE - Unsupported pooling type " + str(fLayerType))

    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        return SOFIE.ROperator_Pool["float"](PoolMode, pool_atrr, fLayerInputName, fLayerOutputName)
    raise RuntimeError("TMVA::SOFIE - Operator Pool does not support input type " + str(fLayerDType))
