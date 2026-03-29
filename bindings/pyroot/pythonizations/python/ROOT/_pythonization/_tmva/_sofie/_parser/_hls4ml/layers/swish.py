def MakeHLSSwish(layer):
    from ROOT.TMVA.Experimental import SOFIE
    finput = layer["layerInput"]
    foutput = layer["layerOutput"]
    fLayerDType = layer["layerDType"]
    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        return SOFIE.ROperator_Swish("float")(finput[0], foutput[0])
    raise RuntimeError("TMVA::SOFIE - Swish does not support input type " + str(fLayerDType))
