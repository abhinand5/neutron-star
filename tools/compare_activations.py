#!/usr/bin/env python3
"""Layer-bisect two activation dumps (PLAN §8 Stage 1 / §10).

Reads the shared "NSAC" record format written by `ns eval --dump-activations` and
`tools/oracle_activations`, matches tensors by name, and reports cosine per layer.
The first layer whose cosine drops is where the bug lives.

usage: tools/compare_activations.py ns_act.bin ref_act.bin [--top 20]
"""
import argparse, math, re, struct, sys


def read_records(path):
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:4] != b"NSAC":
        sys.exit(f"{path}: not an NSAC activation dump")
    (n_rec,) = struct.unpack_from("<I", blob, 4)
    off = 8
    out = {}
    for _ in range(n_rec):
        (nl,) = struct.unpack_from("<I", blob, off); off += 4
        name = blob[off:off + nl].decode("utf-8", "replace"); off += nl
        (ne,) = struct.unpack_from("<Q", blob, off); off += 8
        vals = struct.unpack_from("<%df" % ne, blob, off); off += 4 * ne
        out[name] = vals          # last write wins if a name repeats
    return out


def cosine(a, b):
    d = na = nb = 0.0
    for x, y in zip(a, b):
        d += x * y; na += x * x; nb += y * y
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return d / math.sqrt(na * nb)


def maxdiff(a, b):
    m = 0.0; at = -1
    for i, (x, y) in enumerate(zip(a, b)):
        dd = abs(x - y)
        if dd > m: m, at = dd, i
    return m, at


def layer_of(name):
    m = re.search(r"-(\d+)$", name)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ns"); ap.add_argument("ref")
    ap.add_argument("--quiet", action="store_true", help="only show the divergence onset")
    args = ap.parse_args()

    A, B = read_records(args.ns), read_records(args.ref)
    common = [k for k in A if k in B and len(A[k]) == len(B[k])]
    if not common:
        print("no tensors in common. ns names: %s" % sorted(A)[:8])
        print("                     ref names: %s" % sorted(B)[:8])
        sys.exit(1)
    common.sort(key=lambda n: (layer_of(n), n))

    print("%-24s %8s  %12s  %12s" % ("tensor", "n", "cosine", "max|diff|"))
    first_bad = None
    for name in common:
        c = cosine(A[name], B[name])
        md, at = maxdiff(A[name], B[name])
        flag = ""
        if c < 0.9999 and first_bad is None:
            first_bad = (name, c)
            flag = "   <== FIRST DIVERGENCE"
        if not args.quiet or flag or c < 0.9999:
            print("%-24s %8d  %12.8f  %12.6g @%d%s" % (name, len(A[name]), c, md, at, flag))

    print("\n%d tensors compared (%d in ns, %d in ref)" % (len(common), len(A), len(B)))
    if first_bad:
        print("first divergence: %s  cosine=%.8f" % first_bad)
        print("-> the bug is in the computation that produces this tensor")
    else:
        print("no divergence above the 0.9999 threshold at this position")


main()
