from ROOT.TMVA.Experimental import SOFIE


def MakeHLSGemm(layer):
    finput = layer["layerInput"]
    foutput = layer["layerOutput"]
    fLayerDType = layer["layerDType"]
    fLayerInputName = finput[0]
    fLayerOutputName = foutput[0]
    fWeightNames = layer["layerWeight"]
    fKernelName = fWeightNames[0]
    fBiasName = fWeightNames[1]
    attr_alpha = 1.0
    attr_beta = 1.0
    attr_transA = 0
    attr_transB = 0  # Keras/HLS4ML Dense weights are [in, out]; Y = X @ W
    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        op = SOFIE.ROperator_Gemm["float"](
            attr_alpha, attr_beta, attr_transA, attr_transB, fLayerInputName, fKernelName, fBiasName, fLayerOutputName
        )
        return op
    else:
        raise RuntimeError("TMVA::SOFIE - Operator Gemm does not support input type " + fLayerDType)
