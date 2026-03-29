def MakeHLSReshape(layer):
    from ROOT.TMVA.Experimental import SOFIE
    finput = layer["layerInput"]
    foutput = layer["layerOutput"]
    attributes = layer["layerAttributes"]
    flayername = attributes.get("name", "reshape")
    fOpMode = SOFIE.ReshapeOpMode.Reshape
    fNameData = finput[0]
    fNameOutput = foutput[0]
    fNameShape = flayername + "_shape"
    op = SOFIE.ROperator_Reshape(fOpMode, 0, fNameData, fNameShape, fNameOutput)
    return op
