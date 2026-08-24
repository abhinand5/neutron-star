#!/usr/bin/env python3
"""Run the complete Stage 2 G2a correctness gate reproducibly.

The gate has two independent parts for each blessed GGUF:

* 205 teacher-forced rows across the four committed prompts, compared against
  pinned llama.cpp CPU stepwise and batched controls under D6/D8/D9; and
* one 256-token autoregressive continuation from p1's first 12 tokens, which
  must equal one complete reference path (never a weave between paths).

CPU-oracle outputs are expensive, so they live in explicit reference
directories and are reused only after their exact byte counts are validated.
Pass ``--prepare-oracles`` to create any missing references.  Every invocation
writes the commands, logs, source/oracle identities, and artifact hashes to one
run directory, including when the gate is RED.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
VOCAB = 248_320
GREEDY_TOKENS = 256
PINNED_LLAMA_COMMIT = "3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70"
PROMPT_NAMES = ("p1", "p2", "p3", "p4")
EXPECTED_PROMPT_LENGTHS = {"p1": 31, "p2": 90, "p3": 54, "p4": 30}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str
    size: int
    teacher_reference_name: str
    greedy_reference_name: str


MODELS = {
    "q4": ModelSpec(
        "q4",
        "Qwen3.8-27B-UD-Q4_K_XL.gguf",
        17_559_178_144,
        "parity-205",
        "greedy-256-q4",
    ),
    "q5": ModelSpec(
        "q5",
        "Qwen3.8-27B-UD-Q5_K_XL.gguf",
        20_876_938_144,
        "parity-205-q5",
        "greedy-256-q5",
    ),
}


class GateError(RuntimeError):
    pass


def read_prompts(prompt_dir: Path = ROOT / "tests" / "prompts") -> dict[str, str]:
    prompts = {}
    for name in PROMPT_NAMES:
        path = prompt_dir / (name + ".tokens")
        csv = path.read_text(encoding="utf-8").strip()
        tokens = [piece for piece in csv.split(",") if piece]
        expected = EXPECTED_PROMPT_LENGTHS[name]
        if len(tokens) != expected:
            raise GateError(
                f"{path}: expected {expected} committed tokens, found {len(tokens)}"
            )
        for token in tokens:
            int(token)
        prompts[name] = ",".join(tokens)
    if sum(len(csv.split(",")) for csv in prompts.values()) != 205:
        raise GateError("the committed teacher corpus must contain exactly 205 rows")
    return prompts


def expected_teacher_bytes(prompt_csv: str, vocab: int = VOCAB) -> int:
    return len(prompt_csv.split(",")) * vocab * 4


def expected_greedy_bytes(token_count: int = GREEDY_TOKENS) -> int:
    return token_count * 4


def check_artifact(path: Path, expected_bytes: int, label: str) -> None:
    try:
        actual = path.stat().st_size
    except FileNotFoundError as error:
        raise GateError(f"missing {label}: {path}") from error
    if actual != expected_bytes:
        raise GateError(
            f"{label} has {actual} bytes, expected {expected_bytes}: {path}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def teacher_ns_command(ns: Path, model: Path, prompt: str, output: Path) -> list[str]:
    return [
        str(ns), "gpu-eval", str(model), "--tokens", prompt,
        "--ctx", "128", "--topk", "1", "--dump-logits", str(output),
    ]


def teacher_oracle_command(
    oracle: Path, model: Path, prompt: str, output: Path, batched: bool
) -> list[str]:
    command = [str(oracle), "-m", str(model), "-t", prompt]
    if batched:
        command.append("--batched")
    return [*command, "-o", str(output)]


def greedy_ns_command(ns: Path, model: Path, prompt12: str, output: Path) -> list[str]:
    return [
        str(ns), "gpu-eval", str(model), "--tokens", prompt12,
        "--ctx", "272", "--topk", "1", "--generate", str(GREEDY_TOKENS),
        "--dump-tokens", str(output),
    ]


def greedy_oracle_command(
    oracle: Path, model: Path, prompt12: str, output: Path, batched: bool
) -> list[str]:
    command = [str(oracle), "-m", str(model), "-t", prompt12]
    if batched:
        command.append("--batched")
    return [
        *command, "--generate", str(GREEDY_TOKENS), "--dump-tokens", str(output)
    ]


def teacher_compare_command(
    python: str,
    ns_outputs: dict[str, Path],
    teacher_references: Path,
    model_key: str,
) -> list[str]:
    command = [
        python,
        str(ROOT / "tools" / "compare.py"),
        str(ns_outputs["p1"]),
        str(teacher_references / "p1_ref_step.bin"),
        "--control",
        str(teacher_references / "p1_ref_batch.bin"),
        "--case-label",
        "p1",
    ]
    for name in PROMPT_NAMES[1:]:
        command += [
            "--case", name, str(ns_outputs[name]),
            str(teacher_references / f"{name}_ref_step.bin"),
            str(teacher_references / f"{name}_ref_batch.bin"),
        ]
    return [
        *command,
        "--vocab", str(VOCAB),
        "--guard-side", "ns",
        "--label", f"G2a {model_key.upper()} teacher-forced logits",
    ]


def greedy_compare_command(
    python: str,
    ns_output: Path,
    greedy_references: Path,
    model_key: str,
) -> list[str]:
    return [
        python,
        str(ROOT / "tools" / "compare_tokens.py"),
        str(ns_output),
        str(greedy_references / "p1_ref_step.bin"),
        str(greedy_references / "p1_ref_batch.bin"),
        "--min-tokens", str(GREEDY_TOKENS),
        "--label", f"G2a {model_key.upper()} p1 greedy continuation",
    ]


def unique_run_dir(cache_dir: Path, source_commit: str) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = cache_dir / "runs" / f"{stamp}-{source_commit[:12]}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(str(base) + f"-{suffix}")
        suffix += 1
    return candidate


class Runner:
    def __init__(self, run_dir: Path, manifest: dict, dry_run: bool):
        self.run_dir = run_dir
        self.manifest = manifest
        self.dry_run = dry_run

    def checkpoint(self) -> None:
        if not self.dry_run:
            write_manifest(self.run_dir, self.manifest)

    def run(self, label: str, command: list[str], env: dict[str, str]) -> int:
        log_path = self.run_dir / "logs" / (label + ".log")
        record = {
            "label": label,
            "argv": command,
            "environment": {"OMP_NUM_THREADS": env["OMP_NUM_THREADS"]},
            "log": str(log_path),
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        self.manifest["commands"].append(record)
        self.checkpoint()
        print(f"\n[{label}] {shlex.join(command)}", flush=True)
        if self.dry_run:
            record["exit_code"] = None
            return 0
        log_path.parent.mkdir(parents=True, exist_ok=True)
        process = None
        try:
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                record["pid"] = process.pid
                self.checkpoint()
                assert process.stdout is not None
                for line in process.stdout:
                    sys.stdout.write(line)
                    log.write(line)
                return_code = process.wait()
        except BaseException:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            record["exit_code"] = process.returncode if process is not None else None
            record["interrupted"] = True
            record["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            self.checkpoint()
            raise
        record["exit_code"] = return_code
        record["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        self.checkpoint()
        return return_code


def generate_artifact(
    runner: Runner,
    label: str,
    final_path: Path,
    expected_bytes: int,
    command_builder,
    env: dict[str, str],
) -> None:
    if runner.dry_run:
        runner.run(label, command_builder(final_path), env)
        return
    final_path.parent.mkdir(parents=True, exist_ok=True)
    partial = final_path.with_name(final_path.name + f".partial.{os.getpid()}")
    if partial.exists():
        raise GateError(f"refusing to overwrite stale partial artifact: {partial}")
    try:
        return_code = runner.run(label, command_builder(partial), env)
        if return_code:
            raise GateError(f"{label} failed with exit code {return_code}")
        check_artifact(partial, expected_bytes, label)
        os.replace(partial, final_path)
    except Exception:
        # The command log is the useful failure evidence; never let a partial
        # multi-gigabyte oracle dump masquerade as a reusable cache artifact.
        partial.unlink(missing_ok=True)
        raise


def ensure_oracle_artifact(
    runner: Runner,
    label: str,
    path: Path,
    expected_bytes: int,
    prepare: bool,
    refresh: bool,
    command_builder,
    env: dict[str, str],
) -> None:
    if path.exists() and not refresh:
        check_artifact(path, expected_bytes, label)
        print(f"[{label}] reuse {path} ({expected_bytes} bytes)")
        return
    if not prepare:
        if runner.dry_run:
            print(f"[{label}] missing; add --prepare-oracles to create {path}")
            return
        raise GateError(f"missing {label}: {path} (use --prepare-oracles)")
    generate_artifact(
        runner, label, path, expected_bytes, command_builder, env
    )


def artifact_record(path: Path, with_hash: bool) -> dict:
    record = {"path": str(path), "bytes": path.stat().st_size}
    if with_hash:
        record["sha256"] = sha256_file(path)
    return record


def write_manifest(run_dir: Path, manifest: dict) -> None:
    path = run_dir / "manifest.json"
    temporary = path.with_suffix(".json.partial")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, sort_keys=True)
        output.write("\n")
    os.replace(temporary, path)


def reference_dirs(args, spec: ModelSpec) -> tuple[Path, Path]:
    teacher_override = getattr(args, f"{spec.key}_teacher_references")
    greedy_override = getattr(args, f"{spec.key}_greedy_references")
    legacy_root = Path(args.legacy_cache_root).expanduser().resolve()
    teacher = Path(teacher_override).expanduser().resolve() if teacher_override \
        else legacy_root / spec.teacher_reference_name
    greedy = Path(greedy_override).expanduser().resolve() if greedy_override \
        else legacy_root / spec.greedy_reference_name
    return teacher, greedy


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    result.add_argument(
        "--model-dir", default="~/dev/models/Qwen3.8-27B",
        help="directory containing the two blessed GGUF files",
    )
    result.add_argument("--ns", default=str(ROOT / "build" / "release" / "ns"))
    result.add_argument(
        "--oracle", default=str(ROOT / "build" / "release" / "tools" / "oracle_logits")
    )
    result.add_argument(
        "--llama-dir", default="~/dev/inference-engines/llama.cpp",
        help="pinned llama.cpp checkout used by the CPU oracle",
    )
    result.add_argument(
        "--cache-dir", default="~/.cache/neutron-star/g2a-gate",
        help="run manifests and current GPU outputs",
    )
    result.add_argument(
        "--legacy-cache-root", default="~/.cache/neutron-star",
        help="root holding the preserved parity-205 and greedy-256 references",
    )
    for key in MODELS:
        result.add_argument(f"--{key}-teacher-references")
        result.add_argument(f"--{key}-greedy-references")
    result.add_argument(
        "--prepare-oracles", action="store_true",
        help="generate missing pinned-CPU reference artifacts (slow)",
    )
    result.add_argument(
        "--refresh-oracles", action="store_true",
        help="regenerate reference artifacts; implies --prepare-oracles",
    )
    result.add_argument("--threads", type=int, default=8)
    result.add_argument(
        "--hash-models", action="store_true",
        help="also SHA-256 the 17.6/20.9 GB GGUF files in the manifest",
    )
    result.add_argument(
        "--no-artifact-hashes", action="store_true",
        help="skip SHA-256 of the smaller logit/token evidence artifacts",
    )
    result.add_argument(
        "--dry-run", action="store_true",
        help="print the complete plan without creating files or running engines",
    )
    return result


def require_executable(path: Path, label: str) -> None:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise GateError(f"{label} is not executable: {path}")


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.threads <= 0:
        raise GateError("--threads must be positive")
    args.prepare_oracles |= args.refresh_oracles

    prompts = read_prompts()
    model_dir = Path(args.model_dir).expanduser().resolve()
    ns = Path(args.ns).expanduser().resolve()
    oracle = Path(args.oracle).expanduser().resolve()
    llama_dir = Path(args.llama_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()

    source_commit = git_output(ROOT, "rev-parse", "HEAD")
    source_status = git_output(ROOT, "status", "--porcelain", "--untracked-files=all")
    llama_commit = git_output(llama_dir, "rev-parse", "HEAD")
    llama_status = git_output(llama_dir, "status", "--porcelain", "--untracked-files=all")
    if llama_commit != PINNED_LLAMA_COMMIT:
        raise GateError(
            f"llama.cpp is {llama_commit}; G2a requires pinned {PINNED_LLAMA_COMMIT}"
        )
    if llama_status:
        raise GateError("pinned llama.cpp checkout is dirty; oracle provenance is ambiguous")
    if not args.dry_run:
        require_executable(ns, "ns GPU engine")
        require_executable(oracle, "llama.cpp oracle")

    run_dir = unique_run_dir(cache_dir, source_commit)
    manifest = {
        "schema": 1,
        "gate": "G2a",
        "status": "RUNNING",
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "source": {"commit": source_commit, "dirty": bool(source_status)},
        "oracle": {"llama_commit": llama_commit, "llama_dirty": False},
        "models": {},
        "commands": [],
    }
    if args.dry_run:
        print(f"dry-run: source {source_commit}{'-dirty' if source_status else ''}")
        print(f"dry-run: pinned llama.cpp {llama_commit}")
        print(f"dry-run: would write {run_dir}")
    else:
        run_dir.mkdir(parents=True)
        manifest["source"]["ns"] = artifact_record(ns, True)
        manifest["oracle"]["binary"] = artifact_record(oracle, True)
        write_manifest(run_dir, manifest)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    runner = Runner(run_dir, manifest, args.dry_run)
    gate_green = True

    def interrupt_handler(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt_handler)

    try:
        for key in args.models:
            spec = MODELS[key]
            model = model_dir / spec.filename
            if not args.dry_run:
                check_artifact(model, spec.size, f"blessed {key} GGUF")
            teacher_refs, greedy_refs = reference_dirs(args, spec)
            model_record = {
                "path": str(model),
                "expected_bytes": spec.size,
                "teacher_references": str(teacher_refs),
                "greedy_references": str(greedy_refs),
                "prompt_token_sha256": {
                    name: hashlib.sha256(csv.encode("ascii")).hexdigest()
                    for name, csv in prompts.items()
                },
                "artifacts": {},
            }
            manifest["models"][key] = model_record
            if not args.dry_run:
                stat = model.stat()
                model_record.update({"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
                if args.hash_models:
                    print(f"[{key}-model] SHA-256 {model}", flush=True)
                    model_record["sha256"] = sha256_file(model)

            print(f"\n=== {key.upper()} references ===")
            for name in PROMPT_NAMES:
                expected = expected_teacher_bytes(prompts[name])
                for mode, batched in (("step", False), ("batch", True)):
                    path = teacher_refs / f"{name}_ref_{mode}.bin"
                    label = f"{key}-oracle-teacher-{name}-{mode}"
                    ensure_oracle_artifact(
                        runner, label, path, expected,
                        args.prepare_oracles, args.refresh_oracles,
                        lambda output, p=prompts[name], b=batched:
                            teacher_oracle_command(oracle, model, p, output, b),
                        env,
                    )

            prompt12 = ",".join(prompts["p1"].split(",")[:12])
            for mode, batched in (("step", False), ("batch", True)):
                path = greedy_refs / f"p1_ref_{mode}.bin"
                label = f"{key}-oracle-greedy-p1-{mode}"
                ensure_oracle_artifact(
                    runner, label, path, expected_greedy_bytes(),
                    args.prepare_oracles, args.refresh_oracles,
                    lambda output, b=batched:
                        greedy_oracle_command(oracle, model, prompt12, output, b),
                    env,
                )

            if args.dry_run:
                output_root = run_dir / key
            else:
                output_root = run_dir / key
                output_root.mkdir(parents=True)

            print(f"\n=== {key.upper()} current GPU run ===")
            ns_outputs = {}
            model_execution_ok = True
            for name in PROMPT_NAMES:
                output = output_root / f"{name}_ns_gpu.bin"
                ns_outputs[name] = output
                try:
                    generate_artifact(
                        runner,
                        f"{key}-ns-teacher-{name}",
                        output,
                        expected_teacher_bytes(prompts[name]),
                        lambda partial, p=prompts[name]:
                            teacher_ns_command(ns, model, p, partial),
                        env,
                    )
                except GateError:
                    model_execution_ok = False
                    raise

            greedy_output = output_root / "p1_ns_gpu_greedy256.bin"
            generate_artifact(
                runner,
                f"{key}-ns-greedy-p1",
                greedy_output,
                expected_greedy_bytes(),
                lambda partial: greedy_ns_command(ns, model, prompt12, partial),
                env,
            )

            teacher_rc = runner.run(
                f"{key}-compare-teacher",
                teacher_compare_command(sys.executable, ns_outputs, teacher_refs, key),
                env,
            )
            greedy_rc = runner.run(
                f"{key}-compare-greedy",
                greedy_compare_command(sys.executable, greedy_output, greedy_refs, key),
                env,
            )
            if teacher_rc not in (0, 1) or greedy_rc not in (0, 1):
                raise GateError(
                    f"{key} comparison tool error: teacher={teacher_rc}, "
                    f"greedy={greedy_rc}"
                )
            model_green = model_execution_ok and teacher_rc == 0 and greedy_rc == 0
            model_record["status"] = "GREEN" if model_green else "RED"
            gate_green &= model_green

            if not args.dry_run:
                artifact_paths = [*ns_outputs.values(), greedy_output]
                artifact_paths += [
                    teacher_refs / f"{name}_ref_{mode}.bin"
                    for name in PROMPT_NAMES for mode in ("step", "batch")
                ]
                artifact_paths += [
                    greedy_refs / f"p1_ref_{mode}.bin" for mode in ("step", "batch")
                ]
                model_record["artifacts"] = {
                    str(path): artifact_record(path, not args.no_artifact_hashes)
                    for path in artifact_paths
                }
                write_manifest(run_dir, manifest)
    except KeyboardInterrupt:
        manifest["status"] = "INTERRUPTED"
        manifest["error"] = "interrupted before the G2a workflow completed"
        if not args.dry_run:
            manifest["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            write_manifest(run_dir, manifest)
            print(f"\nG2a workflow INTERRUPTED\nmanifest: {run_dir / 'manifest.json'}",
                  file=sys.stderr)
        return 130
    except (GateError, OSError, subprocess.SubprocessError) as error:
        manifest["status"] = "ERROR"
        manifest["error"] = str(error)
        if not args.dry_run:
            manifest["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            write_manifest(run_dir, manifest)
            print(f"\nG2a workflow ERROR: {error}\nmanifest: {run_dir / 'manifest.json'}",
                  file=sys.stderr)
        else:
            print(f"\nG2a dry-run ERROR: {error}", file=sys.stderr)
        return 2

    if args.dry_run:
        print("\nG2a dry-run complete; no commands were executed")
        return 0

    manifest["status"] = "GREEN" if gate_green else "RED"
    manifest["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_manifest(run_dir, manifest)
    print(f"\nG2a: {manifest['status']}")
    print(f"manifest: {run_dir / 'manifest.json'}")
    return 0 if gate_green else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (GateError, OSError, subprocess.SubprocessError) as error:
        print(f"gate_g2a.py: {error}", file=sys.stderr)
        sys.exit(2)
