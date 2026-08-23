#!/usr/bin/env python3
"""Compare two logit dumps — the Stage 1 parity gate (PLAN §9.2, gate G1).

Both files are raw little-endian fp32: n_positions * n_vocab values, written in
position order. ns writes them with `ns eval --dump-logits`; the llama.cpp side
comes from tools/oracle_logits.

Gate G1, as amended by DECISIONS.md D6 and D8:

  * cosine   — bar is min(configured, control_worst): ns must match the reference
               at least as well as the reference matches itself (D6).
  * top-1    — TRIANGULATED agreement (D8): a position is a miss only when ns's
               argmax differs from BOTH reference configurations. Where the two
               reference paths disagree with each other, the oracle has no answer
               at that position and matching either one is inside its own
               reproducibility band. Threshold stays >= 99.5%.
  * margin   — the anti-rationalisation clause (D8): every remaining miss must be
               a demonstrated near-tie. ns's losing margin on the contested token
               pair must be < 0.25 logits AND smaller than the reference's own
               cross-config drift on those same tokens. A real bug cannot hide
               here: the pre-D7 defect showed a ~14-logit swing.

The RAW (untriangulated) top-1 number is always reported too — drift in it is
still a signal, even when the triangulated number passes.

usage:
  tools/compare.py ns.bin oracle.bin --vocab 248320
  tools/compare.py ns.bin oracle.bin --vocab 248320 --control ref_batch.bin --verbose
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


def margin(v, tok_a, tok_b):
    """Signed margin of tok_a over tok_b in logit vector v."""
    return v[tok_a] - v[tok_b]


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
    ap.add_argument("--guard-side", choices=("ns", "both"), default="both",
                    help="D8 margin guard: 'ns' reads only ns's own conviction (D8's "
                         "literal wording); 'both' (default) takes the worse of ns's "
                         "and the reference's — strict, and the reading that keeps a "
                         "confidently-wrong engine from passing")
    ap.add_argument("--max-margin", type=float, default=0.25,
                    help="D8 margin guard: a residual miss may lose by at most this "
                         "many logits (default 0.25)")
    ap.add_argument("--label", default="G1",
                    help="name for the report line (use CONTROL when diffing two "
                         "reference runs against each other)")
    args = ap.parse_args()

    A = read_logits(args.ns, args.vocab)
    B = read_logits(args.oracle, args.vocab)
    C = read_logits(args.control, args.vocab) if args.control else None
    MARGIN_MAX = args.max_margin
    npos = min(len(A), len(B))
    if len(A) != len(B):
        print(f"warning: {len(A)} positions in ns vs {len(B)} in oracle; comparing {npos}")

    matches = 0          # raw: ns argmax == primary reference argmax
    tri_matches = 0       # triangulated (D8): ns agrees with either reference path
    worst_cos = 1.0
    worst_pos = -1
    rows = []
    misses = []           # positions failing triangulation, with their margins
    for p in range(npos):
        s = stats(A[p], B[p], args.top)
        matches += s["top1_match"]

        # --- D8 triangulation -------------------------------------------------
        ns_tok, ref_tok = s["top1_ns"], s["top1_ref"]
        ctl_tok = topk(C[p], 1)[0] if C is not None and p < len(C) else None
        agrees_ctl = ctl_tok is not None and ns_tok == ctl_tok
        s["top1_ctl"] = ctl_tok
        s["triangulated"] = s["top1_match"] or agrees_ctl
        tri_matches += s["triangulated"]

        if not s["triangulated"]:
            # margin guard: how close was the call, and how much does the
            # reference itself move between its two configurations?
            # D8: "ns's losing margin on the contested token pair" — a near-tie
            # means NEITHER engine held a strong opinion. Measured on both sides:
            #   ref_gap : how far ahead the reference held its choice over ns's
            #   ns_gap  : how far ahead ns held its own choice over the reference's
            # A coin-flip has both ~0. The pre-D7 defect had ns_gap ~14 while
            # ref_gap stayed small — checking only the reference's side would have
            # waved it through, which the synthetic guard test caught.
            ref_gap = margin(B[p], ref_tok, ns_tok)
            ns_gap  = margin(A[p], ns_tok, ref_tok)
            # D8's text says "ns's losing margin", which reads as ns_gap alone.
            # Taken literally that lets a confidently-wrong ns pass whenever the
            # reference is also near a boundary, so the strict reading takes the
            # worse of the two convictions. The two disagree only when the engines
            # disagree about how close the call is — exactly p1 pos 9. Both are
            # reported; --guard-side selects which one gates.
            ns_margin = ns_gap if args.guard_side == "ns" else max(ref_gap, ns_gap)
            drift = 0.0
            if C is not None and p < len(C):
                for t in (ns_tok, ref_tok):
                    drift = max(drift, abs(B[p][t] - C[p][t]))
            misses.append({"pos": p, "ns": ns_tok, "ref": ref_tok, "ctl": ctl_tok,
                           "margin": ns_margin, "drift": drift,
                           "ref_gap": ref_gap, "ns_gap": ns_gap,
                           "ok": ns_margin < MARGIN_MAX and (C is None or ns_margin < max(drift, 1e-9))})
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
    tri_rate = tri_matches / float(npos) if npos else 0.0
    mean_cos = sum(r["cosine"] for r in rows) / float(npos) if npos else 0.0
    mean_overlap = sum(r["topk_overlap"] for r in rows) / float(npos) if npos else 0.0
    greedy_ns = [r["top1_ns"] for r in rows]
    greedy_ref = [r["top1_ref"] for r in rows]

    # D8: the greedy criterion triangulates the same way — a continuation counts as
    # identical if every position matches one of the reference paths.
    greedy_ok = all(r["triangulated"] for r in rows) if C is not None \
        else greedy_ns == greedy_ref

    print("\n=== parity report (%d positions) ===" % npos)
    print("  top-1 raw         %.4f  (%d/%d vs the primary reference)"
          % (top1_rate, matches, npos))
    if C is not None:
        print("  top-1 triangulated %.4f  (%d/%d, D8: miss = disagrees with BOTH refs)"
              % (tri_rate, tri_matches, npos))
    else:
        print("  top-1 triangulated  n/a   (pass --control to enable D8 triangulation)")
    print("  cosine  mean      %.8f" % mean_cos)
    print("  cosine  worst     %.8f at pos %d  (gate >= %.6f)"
          % (worst_cos, worst_pos, args.min_cosine))
    print("  top-%d overlap     %.4f" % (args.top, mean_overlap))
    print("  greedy identical  %s%s" % (greedy_ns == greedy_ref,
          "" if C is None else "   (triangulated: %s)" % greedy_ok))

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

    # --- D8 margin guard ------------------------------------------------------
    guard_ok = True
    if misses:
        print("\n=== residual misses (D8 margin guard) ===")
        print("  every miss must be a near-tie: margin < %.2f logits and below the"
              % MARGIN_MAX)
        print("  reference's own cross-config drift on the contested pair\n")
        for m in misses:
            print("  pos %3d  ns=%-7d ref=%-7d ctl=%-7s  worst conviction %.4f "
                  "(ref held by %.4f, ns held by %.4f)  ref drift %.4f  %s"
                  % (m["pos"], m["ns"], m["ref"],
                     "n/a" if m["ctl"] is None else str(m["ctl"]),
                     m["margin"], m["ref_gap"], m["ns_gap"], m["drift"],
                     "OK" if m["ok"] else "*** NOT A NEAR-TIE ***"))
            if not m["ok"]:
                guard_ok = False
        if not guard_ok:
            print("\n  a miss failed the margin guard — that is a real disagreement,")
            print("  not oracle noise. Bisect it (PLAN §10) before touching the gate.")

    # The gate scores triangulated agreement when a control is supplied (D8);
    # without one there is nothing to triangulate against and raw is all we have.
    scored_top1 = tri_rate if C is not None else top1_rate
    ok = (scored_top1 >= args.min_top1 and worst_cos >= cos_bar
          and greedy_ok and guard_ok)
    print("\n%s: %s" % (args.label, "GREEN" if ok else "RED"))
    if not ok:
        print("  divergence hunt: dump per-layer activations on both sides and bisect —")
        print("  the first layer whose cosine drops is where the bug lives (PLAN §8/§10).")
    return 0 if ok else 1


sys.exit(main())
