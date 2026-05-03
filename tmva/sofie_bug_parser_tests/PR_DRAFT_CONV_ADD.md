# [tmva][sofie] Fix Conv+Add fusion null operator path

Fixes #21978.

In SOFIE ONNX parser, `Conv + Add` fusion was routed through `ParseFuseConvAdd`, but that path returned a null operator in the broken implementation. Since parser flow had already marked the `Add` node as fused, downstream nodes could observe missing tensor type registration and fail at runtime.

This patch implements `ParseFuseConvAdd` so fused Conv+Add creates a valid fused `ROperator_Conv` and registers the fused output tensor type.

## Why this matters

Without this fix, a valid `Conv -> Add -> Relu` graph can fail during parse with:

`TMVA::SOFIE ONNX Parser relu op has input tensoradd_out but its type is not yet registered`

## Changes

- implement fused Conv+Add parser in `tmva/sofie_parsers/src/ParseFuseConvAdd.cxx`
- preserve existing Conv attribute handling and output type registration behavior

## Checklist

- tested changes locally
- updated the docs (not necessary)

Respected @lmoneta and @sanjibansg,

Could you please review this fix?

Also, I have local bug reproduction scripts, but they are currently not in ROOT standard test layout.  
Kindly advise if you would like me to add the regression test in this PR, and in which exact test location you prefer (`test/` vs `roottest` / existing SOFIE parser test area).

With regards,  
Samreedh Bhuyan
