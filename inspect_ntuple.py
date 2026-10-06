#!/usr/bin/env python3
"""
USAGE
    python3 inspect_ntuple.py FILE.root [FILE2.root ...]
    python3 inspect_ntuple.py 'DIR/*.root'          # quotes: let python glob
    python3 inspect_ntuple.py FILE.root --list      # dump every branch name
    python3 inspect_ntuple.py FILE.root --peek Vertex_x --peek EVT_MVA1

EXIT STATUS
    0 = every required branch present (graph_build.py will run)
    1 = something missing (the report says exactly what)
"""

import argparse
import glob
import os
import sys


def required_branches():
    """The branch list graph_build.py actually asks uproot for.

    Imported from the real module rather than copied, so this check cannot
    drift out of date when the feature table changes.
    """
    try:
        import graph_build as gb
        return sorted(gb.ALL_BRANCHES), "graph_build.ALL_BRANCHES"
    except Exception as e:                                   # noqa: BLE001
        print(f"  [warn] could not import graph_build ({e}); "
              f"falling back to a hardcoded list", file=sys.stderr)
        return sorted({
            "Vertex_x", "Vertex_y", "Vertex_z", "Vertex_xErr", "Vertex_yErr",
            "Vertex_zErr", "Vertex_isPV", "Vertex_ntrk", "Vertex_chi2",
            "Vertex_mass", "Vertex_thrust_angle", "Vertex_thrusthemis_emin",
            "Vertex_d2PV", "Vertex_d2PVx", "Vertex_d2PVy", "Vertex_d2PVz",
            "Vertex_d2PVErr", "DV_d0", "DV_z0",
            "MC_Vertex_x", "MC_Vertex_y", "MC_Vertex_z", "MC_Vertex_ntrk",
            "MC_Vertex_PDGmother", "MC_Vertex_PDGgmother",
            "EVT_MVA1", "EVT_NVertex", "EVT_Thrust_Mag",
        }), "hardcoded fallback"


def inspect(path, req, req_src, show_all=False, peek=()):
    import uproot

    print(f"\n{'=' * 74}\n{path}\n{'=' * 74}")
    try:
        f = uproot.open(path)
    except Exception as e:                                   # noqa: BLE001
        print(f"  !! cannot open: {e}")
        return False

    keys = [k.split(";")[0] for k in f.keys()]
    trees = []
    for k in dict.fromkeys(keys):
        try:
            obj = f[k]
            if hasattr(obj, "num_entries"):
                trees.append((k, obj))
        except Exception:                                    # noqa: BLE001
            continue
    print(f"  objects at top level : {list(dict.fromkeys(keys))[:8]}"
          + (" ..." if len(set(keys)) > 8 else ""))
    if not trees:
        print("  !! no TTree found in this file")
        return False
    for name, t in trees:
        print(f"  tree '{name}': {t.num_entries:,} entries, "
              f"{len(t.keys()):,} branches")

    if "events" not in [n for n, _ in trees]:
        print("\n  !! graph_build.py opens the tree called 'events' "
              "(graph_build.py line ~641).")
        print("     This file has no such tree, so it cannot be read as-is.")
        return False

    tree = f["events"]
    have = set(k.split(";")[0] for k in tree.keys())
    missing = [b for b in req if b not in have]
    present = [b for b in req if b in have]

    print(f"\n  checking {len(req)} required branches (from {req_src})")
    print(f"    present : {len(present)}")
    print(f"    MISSING : {len(missing)}")

    if show_all:
        print(f"\n  all {len(have)} branches in 'events':")
        for b in sorted(have):
            print(f"    {b}")

    if missing:
        print("\n  missing branches (first 40):")
        for b in missing[:40]:
            print(f"    {b}")
        if len(missing) > 40:
            print(f"    ... and {len(missing) - 40} more")
        print("\n  VERDICT: graph_build.py will refuse this file.")
        # a short sample of what IS there, to identify the format
        sample = sorted(have)[:12]
        print(f"  what this file does have, for identification: {sample}")
        if any(x in have for x in ("Particle", "ReconstructedParticles",
                                   "EFlowTrack", "MCRecoAssociations")):
            print("\n  This looks like raw EDM4hep / Delphes generation output.")
            print("  It needs FCCAnalyses stage 1 run over it before "
                  "graph_build.py can touch it.")
        return False

    print("\n  VERDICT: all required branches present -- graph_build.py can "
          "read this file.")
    for b in peek:
        if b in have:
            try:
                v = tree[b].array(entry_stop=3, library="np")
                print(f"    {b}[:3] = {v}")
            except Exception as e:                           # noqa: BLE001
                print(f"    {b}: could not read ({e})")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Check a ROOT file against graph_build.py's branch list.")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--list", action="store_true",
                    help="print every branch name in the events tree")
    ap.add_argument("--peek", action="append", default=[],
                    help="print the first few values of this branch")
    ap.add_argument("--max_files", type=int, default=2)
    args = ap.parse_args()

    req, req_src = required_branches()
    files = []
    for p in args.paths:
        files += sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
    if not files:
        raise SystemExit("no files matched")

    ok = True
    for p in files[:args.max_files]:
        ok &= inspect(p, req, req_src, args.list, args.peek)
    if len(files) > args.max_files:
        print(f"\n({len(files)} files matched; checked the first "
              f"{args.max_files}. They are almost certainly identical.)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
