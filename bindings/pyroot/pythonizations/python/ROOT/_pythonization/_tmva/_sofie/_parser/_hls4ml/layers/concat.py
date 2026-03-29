def MakeHLSConcat(layer):
    from ROOT.TMVA.Experimental import SOFIE
    finput = layer["layerInput"]
    foutput = layer["layerOutput"]
    fLayerDType = layer["layerDType"]
    attributes = layer["layerAttributes"]
    input_list = [str(i) for i in finput]
    output_name = str(foutput[0])
    axis = int(attributes.get("axis", 1))
    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        op = SOFIE.ROperator_Concat(input_list, axis, 0, output_name)
        return op
    else:
        raise RuntimeError("TMVA::SOFIE - Operator Concat does not support input type " + fLayerDType)
