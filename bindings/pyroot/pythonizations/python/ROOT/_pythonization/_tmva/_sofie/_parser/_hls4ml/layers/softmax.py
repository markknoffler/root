def MakeHLSSoftmax(layer):
    from ROOT.TMVA.Experimental import SOFIE
    finput = layer["layerInput"]
    foutput = layer["layerOutput"]
    fLayerDType = layer["layerDType"]
    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        return SOFIE.ROperator_Softmax(-1, finput[0], foutput[0], False)
    raise RuntimeError("TMVA::SOFIE - Softmax does not support input type " + str(fLayerDType))
