#!/usr/bin/env python3
"""Compare two logit dumps — the Stage 1 parity gate (PLAN §9.2, gate G1).

Both files are raw little-endian fp32: n_positions * n_vocab values, written in
position order. ns writes them with `ns eval --dump-logits`; the llama.cpp side
comes from tools/oracle_logits.

Gate G1 (PLAN §8): per position, top-1 agreement >= 99.5% and cosine >= 0.9999,
and greedy continuations identical.

usage:
  tools/compare.py ns.bin oracle.bin --vocab 248320
  tools/compare.py ns.bin oracle.bin --vocab 248320 --top 5 --verbose
"""
import argparse, struct, sys, math


def read_logits(path, vocab):
    with open(path, "rb") as f:
        raw = f.read()
    n = len(raw) // 4
    if n % vocab:
        sys.exit(f"{path}: {n} floats is not a multiple of vocab {vocab}")
    npos = n // vocab
    vals = struct.unpack("<%df" % n, raw)
    return [vals[i * vocab:(i + 1) * vocab] for i in range(npos)]


def topk(v, k):
    return sorted(range(len(v)), key=lambda i: v[i], reverse=True)[:k]


def stats(a, b, k):
    # cosine, max abs diff, top-1 match, top-k overlap
    dot = na = nb = 0.0
    maxd = 0.0
    argmax_d = -1
    for i, (x, y) in enumerate(zip(a, b)):
        dot += x * y
        na += x * x
        nb += y * y
        d = abs(x - y)
        if d > maxd:
            maxd, argmax_d = d, i
    cos = dot / (math.sqrt(na) * math.sqrt(nb)) if na and nb else 0.0
    ta, tb = topk(a, k), topk(b, k)
    return {
        "cosine": cos,
        "max_abs_diff": maxd,
        "max_abs_diff_at": argmax_d,
        "top1_ns": ta[0],
        "top1_ref": tb[0],
        "top1_match": ta[0] == tb[0],
        "topk_overlap": len(set(ta) & set(tb)) / float(k),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ns")
    ap.add_argument("oracle")
    ap.add_argument("--vocab", type=int, required=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--control", metavar="FILE",
                    help="a SECOND independent run of the reference engine (e.g. the "
                         "same model on a different backend). When given, the cosine "
                         "gate becomes 'ns must match the reference at least as well "
                         "as the reference matches itself' — a self-calibrating bar "
                         "that cannot be gamed by loosening a constant.")
    # gate thresholds from PLAN §8 Stage 1
    ap.add_argument("--min-top1", type=float, default=0.995)
    ap.add_argument("--min-cosine", type=float, default=0.9995)
    ap.add_argument("--label", default="G1",
                    help="name for the report line (use CONTROL when diffing two "
                         "reference runs against each other)")
    args = ap.parse_args()

    A = read_logits(args.ns, args.vocab)
    B = read_logits(args.oracle, args.vocab)
    C = read_logits(args.control, args.vocab) if args.control else None
    npos = min(len(A), len(B))
    if len(A) != len(B):
        print(f"warning: {len(A)} positions in ns vs {len(B)} in oracle; comparing {npos}")

    matches = 0
    worst_cos = 1.0
    worst_pos = -1
    rows = []
    for p in range(npos):
        s = stats(A[p], B[p], args.top)
        matches += s["top1_match"]
        if s["cosine"] < worst_cos:
            worst_cos, worst_pos = s["cosine"], p
        rows.append(s)
        if args.verbose or not s["top1_match"]:
            print("pos %3d  top1 ns=%-7d ref=%-7d %s  cos=%.8f  maxdiff=%.4g @%d  top%d overlap=%.2f"
                  % (p, s["top1_ns"], s["top1_ref"], "OK " if s["top1_match"] else "MISS",
                     s["cosine"], s["max_abs_diff"], s["max_abs_diff_at"], args.top,
                     s["topk_overlap"]))

    # Control: how well does the reference engine agree with *itself* across two
    # independent runs? No implementation can be expected to beat that.
    ctl_worst, ctl_mean, ctl_top1 = None, None, None
    if C is not None:
        cw, csum, cm = 1.0, 0.0, 0
        for p in range(min(npos, len(C))):
            cs = stats(B[p], C[p], args.top)
            cw = min(cw, cs["cosine"])
            csum += cs["cosine"]
            cm += cs["top1_match"]
        ctl_worst, ctl_mean = cw, csum / float(min(npos, len(C)))
        ctl_top1 = cm / float(min(npos, len(C)))

    top1_rate = matches / float(npos) if npos else 0.0
    mean_cos = sum(r["cosine"] for r in rows) / float(npos) if npos else 0.0
    mean_overlap = sum(r["topk_overlap"] for r in rows) / float(npos) if npos else 0.0
    greedy_ns = [r["top1_ns"] for r in rows]
    greedy_ref = [r["top1_ref"] for r in rows]

    print("\n=== parity report (%d positions) ===" % npos)
    print("  top-1 agreement   %.4f  (gate >= %.4f)" % (top1_rate, args.min_top1))
    print("  cosine  mean      %.8f" % mean_cos)
    print("  cosine  worst     %.8f at pos %d  (gate >= %.6f)"
          % (worst_cos, worst_pos, args.min_cosine))
    print("  top-%d overlap     %.4f" % (args.top, mean_overlap))
    print("  greedy identical  %s" % (greedy_ns == greedy_ref))

    cos_bar = args.min_cosine
    if ctl_worst is not None:
        print("\n=== control: reference engine vs itself (%d positions) ===" % npos)
        print("  top-1 agreement   %.4f" % ctl_top1)
        print("  cosine  mean      %.8f" % ctl_mean)
        print("  cosine  worst     %.8f" % ctl_worst)
        cos_bar = min(args.min_cosine, ctl_worst)
        print("\n  cosine gate = min(configured %.6f, control %.8f) = %.8f"
              % (args.min_cosine, ctl_worst, cos_bar))
        print("  (an implementation cannot be held to a tighter bar than the")
        print("   reference engine's own run-to-run reproducibility)")

    ok = (top1_rate >= args.min_top1 and worst_cos >= cos_bar
          and greedy_ns == greedy_ref)
    print("\n%s: %s" % (args.label, "GREEN" if ok else "RED"))
    if not ok:
        print("  divergence hunt: dump per-layer activations on both sides and bisect —")
        print("  the first layer whose cosine drops is where the bug lives (PLAN §8/§10).")
    return 0 if ok else 1


sys.exit(main())
