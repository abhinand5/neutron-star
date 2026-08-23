#!/usr/bin/env python3
"""Compare logit dumps for the Stage 1 parity gate (PLAN §9.2, G1).

Inputs are raw little-endian fp32 rows with ``n_vocab`` values per position.
The primary reference is llama.cpp CPU stepwise decode; each comparison also
requires the CPU batched control used by DECISIONS.md D6/D8.

D8 changes top-1 scoring in two ways:

* ns agrees when its argmax matches either reference configuration;
* every remaining miss must have an ns-side margin below 0.25 logits and below
  the reference's cross-configuration drift on the contested pair.

The raw agreement with the primary reference remains visible. Multiple
``--case`` triples are processed one at a time so the complete 192+ position
gate can be evaluated without retaining all three multi-gigabyte dumps in RAM.

This tool evaluates teacher-forced logit parity. It deliberately does not call
that a greedy-continuation test: generated tokens must be fed back into each
engine and checked separately for PLAN's continuation criterion.
"""

import argparse
import math
import os
import struct
import sys


MIN_GATE_POSITIONS = 192
MAX_ADMISSIBLE_MARGIN = 0.25


class LogitReader:
    """Sequential reader that holds one vocabulary row in memory at a time."""

    def __init__(self, path, vocab):
        self.path = path
        self.row_struct = struct.Struct("<%df" % vocab)
        size = os.path.getsize(path)
        if size % self.row_struct.size:
            raise ValueError(
                "%s: %d bytes is not a multiple of row size %d"
                % (path, size, self.row_struct.size)
            )
        self.npos = size // self.row_struct.size
        self.file = open(path, "rb")

    def read(self):
        raw = self.file.read(self.row_struct.size)
        if len(raw) != self.row_struct.size:
            raise ValueError("%s: short read" % self.path)
        return self.row_struct.unpack(raw)

    def close(self):
        self.file.close()


def topk(values, k):
    return sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:k]


def margin(values, winner, loser):
    """Positive conviction for ``winner`` over ``loser`` in one logit row."""
    return values[winner] - values[loser]


def stats(lhs, rhs, k):
    dot = norm_lhs = norm_rhs = 0.0
    max_diff = 0.0
    max_diff_at = -1
    for i, (x, y) in enumerate(zip(lhs, rhs)):
        dot += x * y
        norm_lhs += x * x
        norm_rhs += y * y
        diff = abs(x - y)
        if diff > max_diff:
            max_diff, max_diff_at = diff, i
    cosine = dot / (math.sqrt(norm_lhs) * math.sqrt(norm_rhs)) \
        if norm_lhs and norm_rhs else 0.0
    top_lhs, top_rhs = topk(lhs, k), topk(rhs, k)
    return {
        "cosine": cosine,
        "max_abs_diff": max_diff,
        "max_abs_diff_at": max_diff_at,
        "top1_lhs": top_lhs[0],
        "top1_rhs": top_rhs[0],
        "top1_match": top_lhs[0] == top_rhs[0],
        "topk_overlap": len(set(top_lhs) & set(top_rhs)) / float(k),
    }


def empty_summary(label):
    return {
        "label": label,
        "positions": 0,
        "raw_matches": 0,
        "tri_matches": 0,
        "cosine_sum": 0.0,
        "worst_cosine": 1.0,
        "worst_position": -1,
        "overlap_sum": 0.0,
        "control_matches": 0,
        "control_cosine_sum": 0.0,
        "control_worst_cosine": 1.0,
        "misses": [],
    }


def compare_case(label, ns_path, ref_path, control_path, args):
    readers = []
    try:
        ns_reader = LogitReader(ns_path, args.vocab)
        readers.append(ns_reader)
        ref_reader = LogitReader(ref_path, args.vocab)
        readers.append(ref_reader)
        control_reader = LogitReader(control_path, args.vocab)
        readers.append(control_reader)

        lengths = (ns_reader.npos, ref_reader.npos, control_reader.npos)
        if len(set(lengths)) != 1:
            raise ValueError(
                "%s: row-count mismatch: ns=%d ref=%d control=%d"
                % ((label,) + lengths)
            )
        if not ns_reader.npos:
            raise ValueError("%s: empty logit dump" % label)

        summary = empty_summary(label)
        summary["positions"] = ns_reader.npos
        for position in range(ns_reader.npos):
            ns_logits = ns_reader.read()
            ref_logits = ref_reader.read()
            control_logits = control_reader.read()

            primary = stats(ns_logits, ref_logits, args.top)
            control = stats(ref_logits, control_logits, args.top)
            ns_token = primary["top1_lhs"]
            ref_token = primary["top1_rhs"]
            control_token = control["top1_rhs"]
            triangulated = primary["top1_match"] or ns_token == control_token

            summary["raw_matches"] += primary["top1_match"]
            summary["tri_matches"] += triangulated
            summary["cosine_sum"] += primary["cosine"]
            summary["overlap_sum"] += primary["topk_overlap"]
            summary["control_matches"] += control["top1_match"]
            summary["control_cosine_sum"] += control["cosine"]
            summary["control_worst_cosine"] = min(
                summary["control_worst_cosine"], control["cosine"]
            )
            if primary["cosine"] < summary["worst_cosine"]:
                summary["worst_cosine"] = primary["cosine"]
                summary["worst_position"] = position

            if not triangulated:
                # D8's worked p1 example identifies the ns-side 0.0006 gap as
                # "ns's losing margin". The optional strict reading remains a
                # diagnostic, but D9 adopts the literal ns-side quantity.
                ref_gap = margin(ref_logits, ref_token, ns_token)
                ns_gap = margin(ns_logits, ns_token, ref_token)
                guarded_margin = ns_gap if args.guard_side == "ns" \
                    else max(ref_gap, ns_gap)
                drift = max(
                    abs(ref_logits[ns_token] - control_logits[ns_token]),
                    abs(ref_logits[ref_token] - control_logits[ref_token]),
                )
                summary["misses"].append({
                    "position": position,
                    "ns_token": ns_token,
                    "ref_token": ref_token,
                    "control_token": control_token,
                    "guarded_margin": guarded_margin,
                    "ref_gap": ref_gap,
                    "ns_gap": ns_gap,
                    "drift": drift,
                    "admissible": (
                        guarded_margin < MAX_ADMISSIBLE_MARGIN
                        and guarded_margin < drift
                    ),
                })

            if args.verbose or not primary["top1_match"]:
                print(
                    "%s pos %3d  top1 ns=%-7d ref=%-7d ctl=%-7d %s  "
                    "cos=%.8f  maxdiff=%.4g @%d  top%d overlap=%.2f"
                    % (
                        label,
                        position,
                        ns_token,
                        ref_token,
                        control_token,
                        "OK " if triangulated else "MISS",
                        primary["cosine"],
                        primary["max_abs_diff"],
                        primary["max_abs_diff_at"],
                        args.top,
                        primary["topk_overlap"],
                    )
                )
        return summary
    finally:
        for reader in readers:
            reader.close()


def merge_summaries(summaries):
    total = empty_summary("TOTAL")
    offset = 0
    for summary in summaries:
        for key in (
            "positions",
            "raw_matches",
            "tri_matches",
            "cosine_sum",
            "overlap_sum",
            "control_matches",
            "control_cosine_sum",
        ):
            total[key] += summary[key]
        total["control_worst_cosine"] = min(
            total["control_worst_cosine"], summary["control_worst_cosine"]
        )
        if summary["worst_cosine"] < total["worst_cosine"]:
            total["worst_cosine"] = summary["worst_cosine"]
            total["worst_position"] = offset + summary["worst_position"]
        for miss in summary["misses"]:
            total["misses"].append({"case": summary["label"], **miss})
        offset += summary["positions"]
    return total


def rate(numerator, denominator):
    return numerator / float(denominator) if denominator else 0.0


def print_case_table(summaries):
    print("\n=== per-case parity ===")
    print("  %-12s %5s %12s %12s %13s %13s" % (
        "case", "pos", "raw top-1", "tri top-1", "ns worst cos", "ctl worst cos"
    ))
    for summary in summaries:
        npos = summary["positions"]
        print("  %-12s %5d %5d/%-6d %5d/%-6d %13.8f %13.8f" % (
            summary["label"],
            npos,
            summary["raw_matches"],
            npos,
            summary["tri_matches"],
            npos,
            summary["worst_cosine"],
            summary["control_worst_cosine"],
        ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ns")
    parser.add_argument("oracle")
    parser.add_argument("--control", required=True, metavar="FILE",
                        help="second reference configuration required by D6/D8")
    parser.add_argument("--case-label", default="primary",
                        help="label for the positional input triple")
    parser.add_argument(
        "--case",
        action="append",
        nargs=4,
        default=[],
        metavar=("LABEL", "NS", "ORACLE", "CONTROL"),
        help="append another labelled input triple; may be repeated",
    )
    parser.add_argument("--vocab", type=int, required=True)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--min-top1", type=float, default=0.995)
    parser.add_argument("--min-cosine", type=float, default=0.9995)
    parser.add_argument(
        "--guard-side",
        choices=("ns", "both"),
        default="ns",
        help="D8/D9 gate uses 'ns'; 'both' reports the stricter diagnostic reading",
    )
    parser.add_argument("--label", default="G1 logits")
    args = parser.parse_args()

    cases = [(args.case_label, args.ns, args.oracle, args.control)]
    cases.extend(tuple(case) for case in args.case)
    try:
        summaries = [compare_case(*case, args) for case in cases]
    except (OSError, ValueError) as error:
        print("compare.py: %s" % error, file=sys.stderr)
        return 2

    total = merge_summaries(summaries)
    npos = total["positions"]
    raw_rate = rate(total["raw_matches"], npos)
    triangulated_rate = rate(total["tri_matches"], npos)
    mean_cosine = rate(total["cosine_sum"], npos)
    mean_overlap = rate(total["overlap_sum"], npos)
    control_rate = rate(total["control_matches"], npos)
    control_mean_cosine = rate(total["control_cosine_sum"], npos)
    cosine_bar = min(args.min_cosine, total["control_worst_cosine"])

    print_case_table(summaries)
    print("\n=== aggregate parity report (%d positions) ===" % npos)
    print("  top-1 raw          %.4f  (%d/%d vs primary reference)" % (
        raw_rate, total["raw_matches"], npos
    ))
    print("  top-1 triangulated %.4f  (%d/%d, miss = disagrees with both refs)" % (
        triangulated_rate, total["tri_matches"], npos
    ))
    print("  cosine mean        %.8f" % mean_cosine)
    print("  cosine worst       %.8f at aggregate pos %d" % (
        total["worst_cosine"], total["worst_position"]
    ))
    print("  top-%d overlap       %.4f" % (args.top, mean_overlap))
    print("  greedy continuation NOT TESTED (these are teacher-forced logit rows)")

    print("\n=== control: reference engine vs itself ===")
    print("  top-1 agreement     %.4f" % control_rate)
    print("  cosine mean         %.8f" % control_mean_cosine)
    print("  cosine worst        %.8f" % total["control_worst_cosine"])
    print("  cosine gate = min(configured %.6f, control %.8f) = %.8f" % (
        args.min_cosine, total["control_worst_cosine"], cosine_bar
    ))

    guard_ok = True
    if total["misses"]:
        print("\n=== residual misses (D8/D9 margin guard: %s side) ===" % args.guard_side)
        print("  guarded margin must be < %.2f logits and < reference drift" %
              MAX_ADMISSIBLE_MARGIN)
        for miss in total["misses"]:
            status = "OK" if miss["admissible"] else "*** NOT ADMISSIBLE ***"
            print(
                "  %-12s pos %3d  ns=%-7d ref=%-7d ctl=%-7d  guarded %.4f "
                "(ref %.4f, ns %.4f)  drift %.4f  %s"
                % (
                    miss["case"],
                    miss["position"],
                    miss["ns_token"],
                    miss["ref_token"],
                    miss["control_token"],
                    miss["guarded_margin"],
                    miss["ref_gap"],
                    miss["ns_gap"],
                    miss["drift"],
                    status,
                )
            )
            guard_ok &= miss["admissible"]

    sample_ok = npos >= MIN_GATE_POSITIONS
    if not sample_ok:
        print("\n  sample-size gate: %d < required %d positions" % (
            npos, MIN_GATE_POSITIONS
        ))
    ok = (
        sample_ok
        and triangulated_rate >= args.min_top1
        and total["worst_cosine"] >= cosine_bar
        and guard_ok
    )
    print("\n%s: %s" % (args.label, "GREEN" if ok else "RED"))
    if not ok and not guard_ok:
        print("  a residual miss is outside the oracle-noise guard; bisect it (PLAN §10)")
    return 0 if ok else 1


sys.exit(main())
