#!/usr/bin/env python3
"""Check an ns greedy continuation against either D8 reference path."""

import argparse
import os
import struct
import sys


def read_tokens(path):
    size = os.path.getsize(path)
    if size % 4:
        raise ValueError("%s: %d bytes is not raw int32 token data" % (path, size))
    with open(path, "rb") as token_file:
        raw = token_file.read()
    return struct.unpack("<%di" % (size // 4), raw)


def first_difference(lhs, rhs):
    for position, (left, right) in enumerate(zip(lhs, rhs)):
        if left != right:
            return position
    return None if len(lhs) == len(rhs) else min(len(lhs), len(rhs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ns")
    parser.add_argument("reference_stepwise")
    parser.add_argument("reference_batched")
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--label", default="G1 greedy continuation")
    args = parser.parse_args()

    try:
        ns_tokens = read_tokens(args.ns)
        stepwise_tokens = read_tokens(args.reference_stepwise)
        batched_tokens = read_tokens(args.reference_batched)
    except (OSError, ValueError) as error:
        print("compare_tokens.py: %s" % error, file=sys.stderr)
        return 2

    lengths = (len(ns_tokens), len(stepwise_tokens), len(batched_tokens))
    if len(set(lengths)) != 1:
        print("token-count mismatch: ns=%d stepwise=%d batched=%d" % lengths)
        return 1
    enough = len(ns_tokens) >= args.min_tokens
    matches_stepwise = ns_tokens == stepwise_tokens
    matches_batched = ns_tokens == batched_tokens
    references_identical = stepwise_tokens == batched_tokens

    print("tokens                 %d (gate >= %d)" % (len(ns_tokens), args.min_tokens))
    print("reference paths equal  %s" % references_identical)
    print("ns == stepwise         %s" % matches_stepwise)
    print("ns == batched          %s" % matches_batched)
    if not matches_stepwise:
        print("first stepwise mismatch %s" % first_difference(ns_tokens, stepwise_tokens))
    if not matches_batched:
        print("first batched mismatch  %s" % first_difference(ns_tokens, batched_tokens))

    passed = enough and (matches_stepwise or matches_batched)
    print("%s: %s" % (args.label, "GREEN" if passed else "RED"))
    return 0 if passed else 1


sys.exit(main())
