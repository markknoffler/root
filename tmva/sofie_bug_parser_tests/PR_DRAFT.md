# [tmva][sofie] Wrong fusion child lookup after ONNX graph reordering #21948

Fixes #21948.

In SOFIE ONNX parsing, `nodesChildren` is keyed by original ONNX node index, while the parsing loop iterates over execution order slots from `nodesOrder`.

Previously, the parser passed `nodesChildren[i]` to `ParseOperator`.  
For graphs that are not already topologically ordered in file layout, this can pass children for the wrong node and trigger invalid fusion attempts.

This patch fixes the lookup to use the current node id from reordered execution:

`nodesChildren[nodesOrder[i]]`

so fusion decisions are made using the correct child list of the node being parsed.

## Why this matters

This bug can cause runtime parse failures on valid models when reorder is required, e.g. wrong `MatMul` + `Add` pairing.

## Changes

- one-line fix in `tmva/sofie_parsers/src/RModelParser_ONNX.cxx`
- no extra functional changes

## Checklist

- tested changes locally
- updated the docs (not necessary)

Respected @lmoneta and @sanjibansg,

Could you please review this fix?

Also, I have local bug reproduction scripts, but they are currently not in ROOT standard test layout.  
Kindly advise if you would like me to add the regression test in this PR, and in which exact test location you prefer (`test/` vs `roottest` / existing SOFIE parser test area).

With regards,  
Samreedh Bhuyan
