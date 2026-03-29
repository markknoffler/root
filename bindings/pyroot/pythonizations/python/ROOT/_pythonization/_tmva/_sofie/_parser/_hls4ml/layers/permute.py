def MakeHLSPermute(layer):
    from ROOT.TMVA.Experimental import SOFIE

    finput = layer["layerInput"]
    foutput = layer["layerOutput"]
    fLayerDType = layer["layerDType"]
    fLayerInputName = finput[0]
    fLayerOutputName = foutput[0]
    attributes = layer["layerAttributes"]
    dims = attributes.get("dims")

    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        if dims:
            # Ensure batch dimension (0) is included as hls4ml usually provides 
            # dims for the feature dimensions only (starting from 1)
            # Keras Permute(dims=(2,1)) on (H, W) -> (batch, H, W)
            # hls4ml dims will be [2, 1] for a 3D tensor (including batch)
            # We follow the Keras parser logic: add 0 for batch if not present
            if 0 not in dims:
                fAttributePermute = [0] + [int(d) for d in dims]
            else:
                fAttributePermute = [int(d) for d in dims]
            
            op = SOFIE.ROperator_Transpose("float")(
                fAttributePermute, fLayerInputName, fLayerOutputName
            )
        else:
            op = SOFIE.ROperator_Transpose("float")(fLayerInputName, fLayerOutputName)
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator Transpose does not yet support input type " + fLayerDType
        )
