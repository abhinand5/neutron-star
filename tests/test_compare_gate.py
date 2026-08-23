#!/usr/bin/env python3
"""Self-test for the G1 gate logic in tools/compare.py (DECISIONS D6/D8).

The gate decides whether the engine is correct, so the gate itself needs a test.
Three synthetic cases, each a known verdict:

  A. ns disagrees with the primary reference but agrees with the control
     -> D8 triangulation must ACQUIT (the oracle has no answer there).
  B. ns disagrees with both, by a hair, under the reference's own drift
     -> the margin guard must ADMIT it as a coin-flip.
  C. ns disagrees with both by ~14 logits (the pre-D7 defect signature)
     -> the margin guard must REJECT it, whatever the counts say.

Case C is the one that matters: it is the reason the guard exists. An earlier
draft of the guard measured conviction only on the reference's logits and waved
C straight through.

run: python3 tests/test_compare_gate.py
"""
import copy, os, random, struct, subprocess, sys, tempfile

V, N = 64, 8
HERE = os.path.dirname(os.path.abspath(__file__))
COMPARE = os.path.join(HERE, os.pardir, "tools", "compare.py")


def write(path, rows):
    with open(path, "wb") as f:
        for r in rows:
            f.write(struct.pack("<%df" % V, *r))


def build(tmp):
    random.seed(11)
    base = [[random.gauss(0, 3) for _ in range(V)] for _ in range(N)]
    ref = copy.deepcopy(base)
    ctl = [[x + random.gauss(0, 0.05) for x in r] for r in base]
    ns = [[x + random.gauss(0, 0.05) for x in r] for r in base]
    a, b = 5, 9
    # A: ref picks a, control and ns pick b
    ref[3][a], ref[3][b] = 10.00, 9.95
    ctl[3][a], ctl[3][b] = 9.95, 10.00
    ns[3][a], ns[3][b] = 9.95, 10.00
    # B: both refs pick a by 0.01 while drifting 0.06 on those tokens; ns picks b
    ref[5][a], ref[5][b] = 8.00, 7.99
    ctl[5][a], ctl[5][b] = 8.06, 7.95
    ns[5][a], ns[5][b] = 7.99, 8.00
    write(os.path.join(tmp, "ns.bin"), ns)
    write(os.path.join(tmp, "ref.bin"), ref)
    write(os.path.join(tmp, "ctl.bin"), ctl)
    # C: ns is confidently wrong at the same position
    bad = copy.deepcopy(ns)
    bad[5][a], bad[5][b] = -6.0, 8.0
    write(os.path.join(tmp, "ns_bad.bin"), bad)


def run(tmp, ns_file):
    p = subprocess.run([sys.executable, COMPARE,
                        os.path.join(tmp, ns_file), os.path.join(tmp, "ref.bin"),
                        "--control", os.path.join(tmp, "ctl.bin"),
                        "--vocab", str(V), "--top", "3"],
                       capture_output=True, text=True)
    return p.stdout


def main():
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        build(tmp)

        good = run(tmp, "ns.bin")
        bad = run(tmp, "ns_bad.bin")

        checks = [
            ("A: triangulation acquits the control-agreeing position",
             "top-1 raw         0.7500" in good and "0.8750" in good),
            ("B: a genuine near-tie passes the margin guard",
             "NOT A NEAR-TIE" not in good),
            ("C: a ~14-logit disagreement is rejected by the guard",
             "NOT A NEAR-TIE" in bad),
            ("C: and the failure is explained, not silent",
             "real disagreement" in bad),
            ("raw top-1 stays visible in both reports",
             "top-1 raw" in good and "top-1 raw" in bad),
        ]
        for name, ok in checks:
            print("  %-58s %s" % (name, "PASS" if ok else "FAIL"))
            if not ok:
                failures += 1
        if failures:
            print("\n--- good-case output ---\n" + good)
            print("\n--- bad-case output ---\n" + bad)
    print("  OK — gate logic behaves as DECISIONS D8 specifies" if not failures
          else "  %d gate-logic check(s) FAILED" % failures)
    return 1 if failures else 0


sys.exit(main())
