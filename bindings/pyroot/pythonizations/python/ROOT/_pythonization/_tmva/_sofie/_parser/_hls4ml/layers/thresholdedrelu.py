def MakeHLSThresholdedRelu(layer):
    from ROOT.TMVA.Experimental import SOFIE
    finput = layer["layerInput"]
    foutput = layer["layerOutput"]
    fLayerDType = layer["layerDType"]
    fLayerInputName = finput[0]
    fLayerOutputName = foutput[0]
    alpha = float(layer["layerAttributes"].get("theta", 1.0))
    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        op = SOFIE.ROperator_ThresholdedRelu("float")(alpha, fLayerInputName, fLayerOutputName)
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Operator ThresholdedRelu does not support input type " + fLayerDType
        )
