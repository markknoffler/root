# core/exporter.py
# Serializes the parsed result to JSON.
# Weights stored as nested float32 lists — readable by the new C++ ParseFromPython().

import json
import numpy as np
from typing import Dict


def export_json(parsed: Dict, filepath: str) -> None:
    """
    Write parsed model dict to a JSON file compatible with C++ ParseFromPython().

    Weight arrays (numpy) are converted to nested float32 lists.
    The nodeWeights field inside each operator is removed in the JSON output —
    weights live only in the top-level 'initializers' dict (no duplication).
    """
    ops_serializable = []
    for op in parsed["operators"]:
        op_copy = {k: v for k, v in op.items() if k != "nodeWeights"}
        ops_serializable.append(op_copy)

    out = {
        "operators":    ops_serializable,
        "initializers": {
            k: {"shape": list(v.shape),
                "dtype": "Float",
                "data":  v.astype(np.float32).flatten().tolist()}
            for k, v in parsed["initializers"].items()
        },
        "inputs":  parsed["inputs"],
        "outputs": {k: v for k, v in parsed["outputs"].items()},
    }

    with open(filepath, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[SOFIE Parser] Exported → {filepath} "
          f"({len(out['operators'])} operators, "
          f"{len(out['initializers'])} weight tensors)")

