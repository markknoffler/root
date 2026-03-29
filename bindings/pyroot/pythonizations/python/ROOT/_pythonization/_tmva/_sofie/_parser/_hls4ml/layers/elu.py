def MakeHLSELU(layer):
    from ROOT.TMVA.Experimental import SOFIE
    finput = layer["layerInput"]
    foutput = layer["layerOutput"]
    fLayerDType = layer["layerDType"]
    fLayerInputName = finput[0]
    fLayerOutputName = foutput[0]
    attributes = layer["layerAttributes"]
    fAlpha = attributes.get("alpha", 1.0)
    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        op = SOFIE.ROperator_Elu("float")(float(fAlpha), fLayerInputName, fLayerOutputName)
        return op
    else:
        raise RuntimeError("TMVA::SOFIE - Operator Elu does not support input type " + fLayerDType)
