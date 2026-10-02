#!/usr/bin/env python
"""Collect <out>/<method>/results.json into one T/S/C/G table (markdown). Usage: summarise_pii.py <out_dir> [--json]"""
import glob, json, os, sys

ROLES = ("T", "S", "C", "G")


def load(out_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, "*", "results.json"))):
        r = json.load(open(p))
        row = {"method": os.path.basename(os.path.dirname(p)), "time_s": r.get("time_s"), "iters": r.get("iters"),
               "stopped": r.get("stopped_early")}
        for k in ROLES:
            if k in r["before"] and k in r["after"]:
                row[f"{k}_lp0"] = r["before"][k]["lp_mean"]; row[f"{k}_lp1"] = r["after"][k]["lp_mean"]
                row[f"{k}_ex0"] = r["before"][k].get("extract_rate"); row[f"{k}_ex1"] = r["after"][k].get("extract_rate")
        rows.append(row)
    return rows


def table(rows):
    f = lambda v: "-" if v is None else f"{v:+.3f}" if isinstance(v, float) else str(v)
    hdr = ["method"] + [f"Δ{k} lp" for k in ROLES] + [f"{k} extr 0→1" for k in ROLES] + ["iters", "time(s)"]
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        c = [r["method"]]
        for k in ROLES:
            c.append(f(r[f"{k}_lp1"] - r[f"{k}_lp0"]) if f"{k}_lp1" in r else "-")
        for k in ROLES:
            a, b = r.get(f"{k}_ex0"), r.get(f"{k}_ex1")
            c.append("-" if a is None or b is None else f"{a:.2f}→{b:.2f}")
        c += [str(r.get("iters") or "-"), str(r.get("time_s") or "-")]
        out.append("| " + " | ".join(c) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    rows = load(sys.argv[1])
    print(json.dumps(rows, indent=1) if "--json" in sys.argv else table(rows))
