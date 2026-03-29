from ROOT.TMVA.Experimental import SOFIE


def MakeHLSSigmoid(layer):
    finput = layer["layerInput"]
    foutput = layer["layerOutput"]
    fLayerDType = layer["layerDType"]
    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        return SOFIE.ROperator_Sigmoid("float")(finput[0], foutput[0])
    raise RuntimeError("TMVA::SOFIE - Sigmoid does not support input type " + str(fLayerDType))
