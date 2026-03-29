def MakeHLSSeLU(layer):
    from ROOT.TMVA.Experimental import SOFIE
    finput = layer["layerInput"]
    foutput = layer["layerOutput"]
    fLayerDType = layer["layerDType"]
    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        return SOFIE.ROperator_Selu("float")(finput[0], foutput[0])
    raise RuntimeError("TMVA::SOFIE - Selu does not support input type " + str(fLayerDType))
