#!/usr/bin/env python3
"""Self-test for whole-path greedy-continuation triangulation."""

import os
import struct
import subprocess
import sys
import tempfile


HERE = os.path.dirname(os.path.abspath(__file__))
COMPARE = os.path.join(HERE, os.pardir, "tools", "compare_tokens.py")


def write_tokens(path, tokens):
    with open(path, "wb") as token_file:
        token_file.write(struct.pack("<%di" % len(tokens), *tokens))


def run(ns_path, stepwise_path, batched_path):
    return subprocess.run(
        [
            sys.executable,
            COMPARE,
            ns_path,
            stepwise_path,
            batched_path,
            "--min-tokens",
            "4",
        ],
        capture_output=True,
        text=True,
    )


def main():
    with tempfile.TemporaryDirectory() as tmp:
        stepwise_path = os.path.join(tmp, "step.bin")
        batched_path = os.path.join(tmp, "batch.bin")
        ns_path = os.path.join(tmp, "ns.bin")
        write_tokens(stepwise_path, [1, 2, 3, 4])
        write_tokens(batched_path, [1, 9, 8, 7])

        write_tokens(ns_path, [1, 9, 8, 7])
        matches_alternate = run(ns_path, stepwise_path, batched_path)
        write_tokens(ns_path, [1, 9, 3, 7])
        weaves_paths = run(ns_path, stepwise_path, batched_path)

        checks = [
            (
                "a complete match to the batched path passes",
                matches_alternate.returncode == 0
                and "G1 greedy continuation: GREEN" in matches_alternate.stdout,
            ),
            (
                "mixing tokens from incompatible reference contexts fails",
                weaves_paths.returncode == 1
                and "G1 greedy continuation: RED" in weaves_paths.stdout,
            ),
        ]
        failures = 0
        for name, passed in checks:
            print("  %-62s %s" % (name, "PASS" if passed else "FAIL"))
            failures += not passed
        return 1 if failures else 0


sys.exit(main())
