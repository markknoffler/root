from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import unittest
from shutil import which

import onnx


class TestEdgecaseOnnx(unittest.TestCase):
    def _emit_debug_log(self, run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
        payload = {
            "sessionId": "306bc2",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        try:
            with open(
                "/home/mark/Desktop/deep_learning_projects/CERN/.cursor/debug-306bc2.log",
                "a",
                encoding="utf-8",
            ) as fh:
                fh.write(json.dumps(payload) + "\n")
        except OSError:
            pass

    def _resolve_root_executable(self, run_id: str) -> str:
        candidates = []
        env_root_exe = os.environ.get("ROOT_EXECUTABLE")
        if env_root_exe:
            candidates.append(env_root_exe)

        path_root = which("root")
        if path_root:
            candidates.append(path_root)

        rootsys = os.environ.get("ROOTSYS")
        if rootsys:
            candidates.append(os.path.join(rootsys, "bin", "root"))

        candidates.append("/home/mark/root_build_outside/bin/root")
        candidates.append("/home/mark/root_install_outside/bin/root")

        for exe in candidates:
            if exe and os.path.isfile(exe) and os.access(exe, os.X_OK):
                self._emit_debug_log(
                    run_id,
                    "H2",
                    "test_edgecase_onnx.py:_resolve_root_executable",
                    "resolved root executable",
                    {"path": exe},
                )
                return exe

        self._emit_debug_log(
            run_id,
            "H2",
            "test_edgecase_onnx.py:_resolve_root_executable",
            "failed to resolve root executable",
            {"candidates": candidates},
        )
        self.fail(
            "ROOT executable not found. Set ROOT_EXECUTABLE, or source thisroot.sh, "
            "or ensure /home/mark/root_build_outside/bin/root exists."
        )

    def _run_root_parse(self, model_path: str, run_id: str) -> subprocess.CompletedProcess:
        macro = f"""
{{
    gSystem->Load("libROOTTMVASofieParser");
    TMVA::Experimental::SOFIE::RModelParser_ONNX parser;
    try {{
        auto model = parser.Parse("{model_path}", false);
        std::cout << "PARSE_OK" << std::endl;
    }} catch (const std::exception& e) {{
        std::cerr << "PARSE_EXCEPTION: " << e.what() << std::endl;
        throw;
    }}
}}
"""
        here = os.path.dirname(os.path.abspath(__file__))
        with tempfile.NamedTemporaryFile("w", suffix=".C", dir=here, delete=False, encoding="utf-8") as tmp:
            tmp.write(macro)
            macro_path = tmp.name
        self._emit_debug_log(run_id, "H1", "test_edgecase_onnx.py:_run_root_parse", "starting root parse", {"model": model_path})
        root_exe = self._resolve_root_executable(run_id)
        proc = subprocess.run(
            [root_exe, "-l", "-b", "-q", macro_path],
            capture_output=True,
            text=True,
            check=False,
        )
        self._emit_debug_log(
            run_id,
            "H1",
            "test_edgecase_onnx.py:_run_root_parse",
            "root parse completed",
            {"returncode": proc.returncode, "stdout_tail": proc.stdout[-400:], "stderr_tail": proc.stderr[-400:]},
        )
        try:
            os.remove(macro_path)
        except OSError:
            pass
        return proc

    def test_unschedulable_graph_loads_via_protobuf(self):
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "unschedulable_add.onnx")
        self.assertTrue(os.path.isfile(path))
        model = onnx.load(path)
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "Add")

    def test_valid_control_model(self):
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "valid_tiny_add.onnx")
        onnx.checker.check_model(path)

    def test_root_parser_runtime_repro_on_malformed_graph(self):
        here = os.path.dirname(os.path.abspath(__file__))
        bad = os.path.join(here, "unschedulable_add.onnx")
        good = os.path.join(here, "valid_tiny_add.onnx")

        good_proc = self._run_root_parse(good, run_id="pre-fix-good")
        self.assertEqual(
            good_proc.returncode,
            0,
            msg=f"Control model parse failed unexpectedly.\nSTDOUT:\n{good_proc.stdout}\nSTDERR:\n{good_proc.stderr}",
        )

        bad_proc = self._run_root_parse(bad, run_id="pre-fix-bad")
        bad_text = (bad_proc.stdout + "\n" + bad_proc.stderr).lower()
        self.assertNotEqual(
            bad_proc.returncode,
            0,
            msg=(
                "Malformed graph did not fail at runtime. If this is a fixed build, "
                "update this test expectation to a clean parser exception.\n"
                f"STDOUT:\n{bad_proc.stdout}\nSTDERR:\n{bad_proc.stderr}"
            ),
        )
        self.assertTrue(
            ("segmentation" in bad_text)
            or ("stack trace" in bad_text)
            or ("parse_exception" in bad_text)
            or ("cannot find a new node" in bad_text),
            msg=f"Malformed parse failed but without expected signatures.\nSTDOUT:\n{bad_proc.stdout}\nSTDERR:\n{bad_proc.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
