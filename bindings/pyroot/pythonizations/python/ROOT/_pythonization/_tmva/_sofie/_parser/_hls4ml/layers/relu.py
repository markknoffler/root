def MakeHLSReLU(layer):
    from ROOT.TMVA.Experimental import SOFIE
    finput = layer["layerInput"]
    foutput = layer["layerOutput"]
    fLayerDType = layer["layerDType"]
    fLayerInputName = finput[0]
    fLayerOutputName = foutput[0]
    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        op = SOFIE.ROperator_Relu("float")(fLayerInputName, fLayerOutputName)
        return op
    else:
        raise RuntimeError("TMVA::SOFIE - Operator Relu does not support input type " + fLayerDType)
