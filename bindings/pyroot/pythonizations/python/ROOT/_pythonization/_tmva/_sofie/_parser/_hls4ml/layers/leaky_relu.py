from ROOT.TMVA.Experimental import SOFIE


def MakeHLSLeakyRelu(layer):
    finput = layer["layerInput"]
    foutput = layer["layerOutput"]
    fLayerDType = layer["layerDType"]
    attributes = layer["layerAttributes"]
    if "alpha" in attributes:
        f_alpha = float(attributes["alpha"])
    elif "negative_slope" in attributes:
        f_alpha = float(attributes["negative_slope"])
    else:
        f_alpha = 0.2
    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        return SOFIE.ROperator_LeakyRelu("float")(f_alpha, finput[0], foutput[0])
    raise RuntimeError("TMVA::SOFIE - LeakyRelu does not support input type " + str(fLayerDType))
