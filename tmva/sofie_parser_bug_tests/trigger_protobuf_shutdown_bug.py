from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from shutil import which

from generate_malformed_onnx import make_valid_tiny_add


def _resolve_root() -> str:
    env_exe = os.environ.get("ROOT_EXECUTABLE")
    if env_exe and os.path.isfile(env_exe) and os.access(env_exe, os.X_OK):
        return env_exe

    p = which("root")
    if p and os.path.isfile(p) and os.access(p, os.X_OK):
        return p

    rootsys = os.environ.get("ROOTSYS")
    if rootsys:
        cand = os.path.join(rootsys, "bin", "root")
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand

    for cand in (
        "/home/mark/root_build_outside/bin/root",
        "/home/mark/root_install_outside/bin/root",
    ):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand

    raise FileNotFoundError("Could not find ROOT executable; set ROOT_EXECUTABLE or adjust paths in this script.")


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    model_path = make_valid_tiny_add(here)

    macro = f"""
{{
   gSystem->Load("libROOTTMVASofieParser");
   TMVA::Experimental::SOFIE::RModelParser_ONNX p;
   p.CheckModel("{model_path}", false);
   p.CheckModel("{model_path}", false);
}}
"""
    root_exe = _resolve_root()

    with tempfile.NamedTemporaryFile("w", suffix=".C", dir=here, delete=False, encoding="utf-8") as tmp:
        tmp.write(macro)
        macro_path = tmp.name

    proc = subprocess.run([root_exe, "-l", "-b", "-q", macro_path])
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())

