def MakeHLSBatchNorm(layer):
    from ROOT.TMVA.Experimental import SOFIE
    # build BatchNorm op
    finput = layer["layerInput"]
    foutput = layer["layerOutput"]
    attributes = layer["layerAttributes"]
    fLayerDType = layer["layerDType"]
    fNX = str(finput[0])
    fNY = str(foutput[0])
    fWeightNames = layer["layerWeight"]
    fNScale = fWeightNames[0]
    fNB = fWeightNames[1]
    fNMean = fWeightNames[2]
    fNVar = fWeightNames[3]
    epsilon = float(attributes.get("epsilon", 1e-3))
    momentum = float(attributes.get("momentum", 0.99))
    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        return SOFIE.ROperator_BatchNormalization("float")(
            epsilon, momentum, 0, fNX, fNScale, fNB, fNMean, fNVar, fNY
        )
    raise RuntimeError(
        "TMVA::SOFIE - Operator BatchNormalization does not support input type " + str(fLayerDType)
    )
