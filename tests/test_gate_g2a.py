#!/usr/bin/env python3
"""Pure planning/validation checks for tools/gate_g2a.py."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("gate_g2a", ROOT / "tools" / "gate_g2a.py")
gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


prompts = gate.read_prompts()
assert {name: len(csv.split(",")) for name, csv in prompts.items()} == {
    "p1": 31,
    "p2": 90,
    "p3": 54,
    "p4": 30,
}
assert sum(len(csv.split(",")) for csv in prompts.values()) == 205
assert gate.expected_teacher_bytes(prompts["p2"]) == 89_395_200
assert gate.expected_greedy_bytes() == 1024

ns = Path("/opt/ns")
oracle = Path("/opt/oracle_logits")
model = Path("/models/q5.gguf")
output = Path("/evidence/out.bin")
prompt12 = ",".join(prompts["p1"].split(",")[:12])

teacher = gate.teacher_ns_command(ns, model, prompts["p1"], output)
assert teacher[:3] == ["/opt/ns", "gpu-eval", "/models/q5.gguf"]
assert teacher[teacher.index("--ctx") + 1] == "128"
assert teacher[-2:] == ["--dump-logits", "/evidence/out.bin"]

step = gate.teacher_oracle_command(oracle, model, prompts["p1"], output, False)
batch = gate.teacher_oracle_command(oracle, model, prompts["p1"], output, True)
assert "--batched" not in step
assert "--batched" in batch

greedy = gate.greedy_ns_command(ns, model, prompt12, output)
assert greedy[greedy.index("--ctx") + 1] == "272"
assert greedy[greedy.index("--generate") + 1] == "256"
assert greedy[-2:] == ["--dump-tokens", "/evidence/out.bin"]

ns_outputs = {name: Path(f"/{name}_ns.bin") for name in gate.PROMPT_NAMES}
compare = gate.teacher_compare_command("python3", ns_outputs, Path("/refs"), "q5")
assert compare.count("--case") == 3
assert compare[compare.index("--vocab") + 1] == "248320"
assert compare[compare.index("--guard-side") + 1] == "ns"

token_compare = gate.greedy_compare_command("python3", output, Path("/greedy"), "q5")
assert token_compare[token_compare.index("--min-tokens") + 1] == "256"

with tempfile.TemporaryDirectory() as directory:
    artifact = Path(directory) / "tokens.bin"
    artifact.write_bytes(b"\0" * 1024)
    gate.check_artifact(artifact, 1024, "test artifact")
    try:
        gate.check_artifact(artifact, 1020, "test artifact")
    except gate.GateError as error:
        assert "expected 1020" in str(error)
    else:
        raise AssertionError("wrong-sized gate artifact was accepted")

with tempfile.TemporaryDirectory() as directory:
    run_dir = Path(directory)
    manifest = {"commands": []}
    runner = gate.Runner(run_dir, manifest, False)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "3"
    assert runner.run("checkpoint-test", [sys.executable, "-c", "print('ok')"], env) == 0
    checkpoint = json.loads((run_dir / "manifest.json").read_text())
    assert checkpoint["commands"][0]["exit_code"] == 0
    assert checkpoint["commands"][0]["environment"] == {"OMP_NUM_THREADS": "3"}
    assert "started_at_utc" in checkpoint["commands"][0]
    assert "finished_at_utc" in checkpoint["commands"][0]

print("gate_g2a workflow planning: PASS")
