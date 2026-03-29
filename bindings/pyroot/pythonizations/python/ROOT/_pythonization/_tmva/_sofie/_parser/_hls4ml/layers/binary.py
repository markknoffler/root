from ROOT.TMVA.Experimental import SOFIE


def MakeHLSBinary(layer):
    input_names = layer["layerInput"]
    output = layer["layerOutput"]
    fLayerType = layer["layerType"]
    fLayerDType = layer["layerDType"]
    fX1 = input_names[0]
    fX2 = input_names[1]
    fY = output[0]
    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        if fLayerType == "Add":
            return SOFIE.ROperator_BasicBinary(float, SOFIE.EBasicBinaryOperator.Add)(fX1, fX2, fY)
        if fLayerType == "Subtract":
            return SOFIE.ROperator_BasicBinary(float, SOFIE.EBasicBinaryOperator.Sub)(fX1, fX2, fY)
        return SOFIE.ROperator_BasicBinary(float, SOFIE.EBasicBinaryOperator.Mul)(fX1, fX2, fY)
    raise RuntimeError("TMVA::SOFIE - BasicBinary does not support input type " + str(fLayerDType))
