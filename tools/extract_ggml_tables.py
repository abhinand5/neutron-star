#!/usr/bin/env python3
"""Extract the constant tables ns needs from llama.cpp's ggml-common.h.

ns links no ggml at runtime (PLAN §3.4), so these tables must live in our source
tree — but transcribing a 512-entry grid by hand is how you get a silent one-entry
bug that shows up as garbage tokens six stages later. This script copies them
verbatim and records a checksum so a future session can re-verify against the
pinned commit.

usage: python3 tools/extract_ggml_tables.py [path/to/ggml-common.h] > src/quants_tables.h
"""
import hashlib, re, subprocess, sys, os

DEFAULT = os.path.expanduser("~/dev/inference-engines/llama.cpp/ggml/src/ggml-common.h")
WANT = [("int8_t", "kvalues_iq4nl", 16), ("uint8_t", "kmask_iq2xs", 8),
        ("uint32_t", "iq3s_grid", 512)]

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    src = open(path).read()
    try:
        commit = subprocess.check_output(
            ["git", "-C", os.path.dirname(path), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "unknown"

    out = []
    out.append("// ===========================================================================")
    out.append("// src/quants_tables.h — GENERATED, DO NOT EDIT BY HAND.")
    out.append("//")
    out.append("// Regenerate with:")
    out.append("//   python3 tools/extract_ggml_tables.py > src/quants_tables.h")
    out.append("//")
    out.append("// Source: %s" % path)
    out.append("// llama.cpp commit: %s (PLAN §12 pins 3cb7ffb1a)" % commit)
    out.append("// ===========================================================================")
    out.append("#pragma once")
    out.append("")
    out.append("#include <cstdint>")
    out.append("")
    out.append("namespace ns {")
    out.append("")

    for ctype, name, size in WANT:
        m = re.search(r"GGML_TABLE_BEGIN\(%s,\s*%s,\s*%d\)(.*?)GGML_TABLE_END\(\)"
                      % (ctype, name, size), src, re.S)
        if not m:
            sys.exit("could not find table %s in %s" % (name, path))
        body = m.group(1)
        vals = [v.strip() for v in body.replace("\n", " ").split(",") if v.strip()]
        if len(vals) != size:
            sys.exit("table %s: parsed %d values, header declares %d"
                     % (name, len(vals), size))
        digest = hashlib.sha256(",".join(vals).encode()).hexdigest()[:16]
        out.append("// %s[%d] — sha256(values)[:16] = %s" % (name, size, digest))
        out.append("static constexpr %s %s[%d] = {" % (ctype, name, size))
        per_line = 8 if ctype == "uint32_t" else 16
        for i in range(0, size, per_line):
            out.append("    " + ", ".join(vals[i:i + per_line]) + ",")
        out.append("};")
        out.append("")

    out.append("}  // namespace ns")
    print("\n".join(out))

main()
