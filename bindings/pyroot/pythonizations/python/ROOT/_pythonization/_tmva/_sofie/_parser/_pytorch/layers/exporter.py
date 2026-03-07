import json
import numpy as np


def export_json(parsed, filepath):
    ops_serializable = []
    for op in parsed["operators"]:
        op_copy = {k: v for k, v in op.items() if k != "nodeWeights"}
        ops_serializable.append(op_copy)

    out = {
        "operators": ops_serializable,
        "initializers": {
            k: {
                "shape": list(v.shape),
                "dtype": "Float",
                "data":  v.astype(np.float32).flatten().tolist(),
            }
            for k, v in parsed["initializers"].items()
        },
        "inputs":  parsed["inputs"],
        "outputs": parsed["outputs"],
    }

    with open(filepath, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[SOFIE Parser] Exported → {filepath} "
          f"({len(out['operators'])} operators, "
          f"{len(out['initializers'])} weight tensors)")
