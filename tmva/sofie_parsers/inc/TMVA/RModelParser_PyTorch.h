// @(#)root/tmva/pymva $Id$
// Author: Sanjiban Sengupta, 2021

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 *                                             *
 *                                                                                *
 * Description:                                                                   *
 *      Functionality for parsing a saved PyTorch .PT model into RModel object    *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Sanjiban Sengupta <sanjiban.sg@gmail.com>                                 *
 *                                                                                *
 * Copyright (c) 2021:                                                            *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (see tmva/doc/LICENSE)                                          *
 **********************************************************************************/


#ifndef TMVA_SOFIE_RMODELPARSER_PYTORCH
#define TMVA_SOFIE_RMODELPARSER_PYTORCH

#include "TMVA/RModel.hxx"
#include "TMVA/SOFIE_common.hxx"
#include "TMVA/Types.h"
#include "TMVA/OperatorList.hxx"

#include "Rtypes.h"
#include "TString.h"


namespace TMVA::Experimental::SOFIE::PyTorch {

/// Parser function for translating PyTorch .pt model into a RModel object.
/// Accepts the file location of a PyTorch model, shapes and data-types of input tensors
/// and returns the equivalent RModel object.
RModel Parse(std::string filepath, std::vector<std::vector<size_t>> inputShapes, std::vector<ETensorType> dtype);

/// Overloaded Parser function for translating PyTorch .pt model into a RModel object.
/// Accepts the file location of a PyTorch model and the shapes of input tensors.
/// Builds the vector of data-types for input tensors and calls the `Parse()` function to
/// return the equivalent RModel object.
RModel Parse(std::string filepath, std::vector<std::vector<size_t>> inputShapes);

/// Parse a JSON file produced by the Python SOFIEPyTorchParser into an RModel object.
/// Replaces the broken _model_to_graph ONNX path for modern PyTorch (>= 2.0).
RModel ParseFromPython(std::string jsonFilePath, std::vector<std::vector<size_t>> inputShapes, std::vector<ETensorType> inputDTypes);

/// Convenience overload — defaults all input types to FLOAT.
RModel ParseFromPython(std::string jsonFilePath, std::vector<std::vector<size_t>> inputShapes);

} // namespace TMVA::Experimental::SOFIE::PyTorch

#endif //TMVA_PYMVA_RMODELPARSER_PYTORCH
