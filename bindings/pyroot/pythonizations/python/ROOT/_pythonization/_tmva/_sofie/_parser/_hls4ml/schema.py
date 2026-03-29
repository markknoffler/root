from __future__ import annotations

from typing import Any, Dict, List, MutableMapping, Optional


def sofie_layer_dict(canonical: MutableMapping[str, Any]) -> Dict[str, Any]:
    # remove schema-only keys before creating operators
    out: Dict[str, Any] = {
        "layerType": canonical["layerType"],
        "layerInput": list(canonical["layerInput"]),
        "layerOutput": list(canonical["layerOutput"]),
        "layerDType": canonical.get("layerDType", "float32"),
        "layerAttributes": dict(canonical.get("layerAttributes", {})),
    }
    lw = canonical.get("layerWeight")
    if lw is not None:
        out["layerWeight"] = list(lw)
    if "channels_last" in canonical:
        out["channels_last"] = bool(canonical["channels_last"])
    return out


def canonical_layer_keys() -> List[str]:
    return [
        "layerType",
        "layerInput",
        "layerOutput",
        "layerDType",
        "layerAttributes",
        "layerWeight",
        "initialisers",
        "precision",
        "name",
        "class_name",
    ]
