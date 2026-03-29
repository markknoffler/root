from ROOT.TMVA.Experimental import SOFIE


def MakeHLSTanh(layer):
    finput = layer["layerInput"]
    foutput = layer["layerOutput"]
    fLayerDType = layer["layerDType"]
    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        return SOFIE.ROperator_Tanh("float")(finput[0], foutput[0])
    raise RuntimeError("TMVA::SOFIE - Tanh does not support input type " + str(fLayerDType))
