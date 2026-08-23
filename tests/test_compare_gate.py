#!/usr/bin/env python3
"""Self-test for the D6/D8/D9 logit-gate logic in tools/compare.py."""

import copy
import os
import random
import struct
import subprocess
import sys
import tempfile


VOCAB_SIZE = 64
GATE_POSITIONS = 200
HERE = os.path.dirname(os.path.abspath(__file__))
COMPARE = os.path.join(HERE, os.pardir, "tools", "compare.py")


def write_logits(path, rows):
    with open(path, "wb") as output:
        for row in rows:
            output.write(struct.pack("<%df" % VOCAB_SIZE, *row))


def build_fixtures(tmp):
    random.seed(11)
    base = [
        [random.gauss(0, 3) for _ in range(VOCAB_SIZE)]
        for _ in range(GATE_POSITIONS)
    ]
    reference = copy.deepcopy(base)
    control = copy.deepcopy(base)
    neutron_star = copy.deepcopy(base)
    reference_token, ns_token = 5, 9

    # Triangulation: primary picks reference_token; control and ns pick ns_token.
    reference[3][reference_token], reference[3][ns_token] = 10.00, 9.95
    control[3][reference_token], control[3][ns_token] = 9.95, 10.00
    neutron_star[3][reference_token], neutron_star[3][ns_token] = 9.95, 10.00

    # D8's p1 shape: both refs prefer reference_token confidently, while ns is a
    # 0.0006 near-tie and the reference paths drift by 0.06 on the pair.
    reference[5][reference_token], reference[5][ns_token] = 10.20, 10.00
    control[5][reference_token], control[5][ns_token] = 10.14, 10.04
    neutron_star[5][reference_token], neutron_star[5][ns_token] = 9.9994, 10.00

    confidently_wrong = copy.deepcopy(neutron_star)
    confidently_wrong[5][reference_token], confidently_wrong[5][ns_token] = -4.0, 10.0

    paths = {}
    for name, rows in (
        ("ns", neutron_star),
        ("ns_bad", confidently_wrong),
        ("ref", reference),
        ("control", control),
    ):
        paths[name] = os.path.join(tmp, "%s.bin" % name)
        write_logits(paths[name], rows)
    return paths


def run_compare(paths, ns_name="ns", *extra_args):
    command = [
        sys.executable,
        COMPARE,
        paths[ns_name],
        paths["ref"],
        "--control",
        paths["control"],
        "--vocab",
        str(VOCAB_SIZE),
        "--top",
        "3",
        "--min-cosine",
        "0",
    ]
    command.extend(extra_args)
    return subprocess.run(command, capture_output=True, text=True)


def main():
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        paths = build_fixtures(tmp)
        adopted = run_compare(paths)
        strict = run_compare(paths, "ns", "--guard-side", "both")
        bad = run_compare(paths, "ns_bad")

        checks = [
            (
                "triangulation + one admissible miss yields a GREEN 199/200 gate",
                adopted.returncode == 0
                and "top-1 triangulated 0.9950  (199/200" in adopted.stdout
                and "G1 logits: GREEN" in adopted.stdout,
            ),
            (
                "D9 defaults to ns's 0.0006 margin",
                "guarded 0.0006" in adopted.stdout
                and "NOT ADMISSIBLE" not in adopted.stdout,
            ),
            (
                "the optional both-sides diagnostic rejects the 0.20 ref gap",
                strict.returncode == 1
                and "guarded 0.2000" in strict.stdout
                and "NOT ADMISSIBLE" in strict.stdout,
            ),
            (
                "a confidently-wrong ns is rejected by the adopted guard",
                bad.returncode == 1
                and "guarded 14.0000" in bad.stdout
                and "NOT ADMISSIBLE" in bad.stdout,
            ),
            (
                "raw top-1 remains visible",
                "top-1 raw" in adopted.stdout and "top-1 raw" in bad.stdout,
            ),
            (
                "teacher-forced rows are not mislabeled as a greedy continuation",
                "greedy continuation NOT TESTED" in adopted.stdout,
            ),
        ]
        for name, passed in checks:
            print("  %-70s %s" % (name, "PASS" if passed else "FAIL"))
            if not passed:
                failures += 1
        if failures:
            print("\n--- adopted output ---\n" + adopted.stdout + adopted.stderr)
            print("\n--- strict output ---\n" + strict.stdout + strict.stderr)
            print("\n--- bad output ---\n" + bad.stdout + bad.stderr)

    if failures:
        print("  %d gate-logic check(s) FAILED" % failures)
    else:
        print("  OK — the executable verdict matches DECISIONS D8/D9")
    return 1 if failures else 0


sys.exit(main())
