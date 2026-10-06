#!/usr/bin/env python3
"""Evaluate an already-trained model on held-out probe channels.

WHAT THIS IS FOR
----------------
Everything measured so far is IN-DISTRIBUTION: the test split contains the same
production channels the model trained on. The probe channels (Bd2DTauNu,
Bd2DstTauNu, ...) were deliberately excluded from training by graph_build.py, so
running on them asks a different and harder question: does the routing structure
transfer to parent hadrons the model has never seen?

There is no retraining here. The checkpoint is loaded as-is and run forward.

WHY IT WRITES A FAKE RUN DIRECTORY
----------------------------------
mgt_report.py and attention_maps.py both expect a run directory containing
config.json and routing_test.npz. Rather than teach them a new layout, this
writes exactly that shape into <run>/probe_<name>/, so every existing analysis
script works on the probes with no changes:

    python3 mgt_report.py     --run $R/n8_k2_s0_masked/probe_Bd2DTauNu
    python3 attention_maps.py --run $R/n8_k2_s0_masked/probe_Bd2DTauNu --fig 8

STANDARDISATION
---------------
The probe features MUST be standardised with the ORIGINAL run's statistics, not
statistics recomputed on the probe sample. Otherwise every feature is re-centred
to the probe channel's own mean and the comparison is meaningless -- the model
would be seeing inputs on a scale it never trained on. This script therefore
copies the "stats" block verbatim from the training stats.json and only swaps
the file list. It refuses to run if the feature names disagree.

  python3 probe_eval.py --run RUNDIR --probe_shards DIR --channel Bd2DTauNu
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch

from dataset import GraphDataset, make_loader
from model import MoEGraphTransformer
from train import capture


def build_probe_stats(train_stats_path, probe_shards, channels, out_path):
    """Clone the training stats.json, replacing the split with probe shards.

    Pass SEVERAL channels to get them all into one routing npz. That is the
    useful mode: mgt_report.py's per-channel tables then compare them directly,
    which is the whole point when the question is "do these two decays route
    differently from each other". GraphDataset numbers channels by position in
    this split dict, so they stay separable even though graph_build.py stamps
    every probe shard with channel_id = -1.
    """
    st = json.load(open(train_stats_path))
    shard_dir = os.path.abspath(probe_shards)
    split = {}
    for channel in channels:
        sub = os.path.join(shard_dir, channel)
        cand = ([os.path.join(channel, f) for f in sorted(os.listdir(sub))
                 if f.endswith(".pt")] if os.path.isdir(sub) else [])
        if not cand:
            # shards may sit flat in the directory rather than in a subfolder
            cand = [f for f in sorted(os.listdir(shard_dir))
                    if f.endswith(".pt") and channel in f]
        if not cand:
            raise SystemExit(f"no .pt shards for channel {channel} under "
                             f"{shard_dir}")
        probe0 = torch.load(os.path.join(shard_dir, cand[0]),
                            weights_only=False)
        nf = probe0["node_feats"].shape[-1]
        if nf != len(st["feature_names"]):
            raise SystemExit(
                f"feature count mismatch on {channel}: shard has {nf}, "
                f"training stats has {len(st['feature_names'])}. Those shards "
                f"came from a different graph_build.py; rebuild them.")
        # eval only: train and val are deliberately empty
        split[channel] = {"train": [], "val": [], "test": cand}
        print(f"  {channel}: {len(cand)} shard(s)", flush=True)

    st["shard_dir"] = shard_dir
    st["split"] = split
    json.dump(st, open(out_path, "w"))
    print("  standardisation copied verbatim from the training run", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a trained run directory")
    ap.add_argument("--probe_shards", required=True,
                    help="directory holding the probe shards")
    ap.add_argument("--channel", required=True, nargs="+",
                    help="one or more probe channels. Pass several to put them "
                         "in ONE npz so mgt_report.py compares them side by "
                         "side, e.g. --channel Bd2DstTauNu Bd2Dst3Pi")
    ap.add_argument("--tag", default=None,
                    help="output subdir name; defaults to channels joined "
                         "by '_vs_'")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_events", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--mask_evt", action="store_true",
                    help="Remove the EVT node from the graph AT INFERENCE, "
                         "using the same trained weights. This is an occlusion "
                         "test: the experts are identical to the unmasked run, "
                         "so any routing change is caused by the node's "
                         "absence alone -- unlike comparing against a "
                         "separately trained evt_mode=mlp run, where the "
                         "experts are different objects and e3 does not mean "
                         "the same thing. Note the model never saw graphs "
                         "without an EVT node during training, so this is an "
                         "off-distribution input by construction; that is the "
                         "point of an occlusion test, but say it out loud.")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.run, "config.json")))
    suffix = "_noEVT" if "--mask_evt" in os.sys.argv else ""
    tag = args.tag or "_vs_".join(args.channel)
    outdir = os.path.join(args.run, f"probe_{tag}{suffix}")
    os.makedirs(outdir, exist_ok=True)
    dev = torch.device(args.device or
                       ("cuda" if torch.cuda.is_available() else "cpu"))

    ps = build_probe_stats(cfg["stats"], args.probe_shards, args.channel,
                           os.path.join(outdir, "probe_stats.json"))

    ds = GraphDataset(ps, split="test", shuffle_nodes=False,
                      mask_features=cfg.get("mask_features") or None,
                      evt_mode=cfg.get("evt_mode", "node"), verbose=True)

    if args.mask_evt:
        from graph_build import NODE_EVT, NODE_PAD, ROLE_NOT_A_VERTEX
        ev = (ds.ntype == NODE_EVT) & ds.mask
        n = int(ev.sum())
        if not n:
            raise SystemExit("--mask_evt: this run has no EVT node in the "
                             "graph (evt_mode is already pool/mlp)")
        # unset the mask so attention cannot reach it, and zero the row so no
        # stale values survive. Edge feature 5 (touches EVT) is built in
        # collate from node_type, so it goes to zero automatically.
        ds.mask[ev] = False
        ds.feats[ev] = 0.0
        ds.aux[ev] = 0.0
        ds.ntype[ev] = NODE_PAD
        ds.role[ev] = ROLE_NOT_A_VERTEX
        print(f"  --mask_evt: removed {n:,} EVT nodes from the graph "
              f"(weights unchanged)", flush=True)
    loader = make_loader(ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=2)

    model = MoEGraphTransformer(
        len(ds.feature_names), d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], d_ff=cfg.get("d_ff"),
        n_experts=cfg["n_experts"], k=cfg["k"], dropout=cfg["dropout"],
        use_moe=bool(cfg.get("use_moe", 1)), n_global=ds.n_global).to(dev)
    sd = torch.load(os.path.join(args.run, "best.pt"), map_location=dev,
                    weights_only=False)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    print(f"  loaded {args.run}/best.pt  "
          f"({cfg['n_parameters']:,} params, "
          f"evt_mode={cfg.get('evt_mode', 'node')})", flush=True)

    preds, rt = capture(model, loader, dev, max_events=args.max_events)

    # write in the layout the analysis scripts expect
    np.savez_compressed(os.path.join(outdir, "routing_test.npz"), **rt)
    np.savez_compressed(os.path.join(outdir, "preds_test.npz"), **preds)
    pcfg = dict(cfg)
    pcfg["stats"] = ps
    pcfg["probe_channel"] = args.channel
    pcfg["parent_run"] = os.path.abspath(args.run)
    json.dump(pcfg, open(os.path.join(outdir, "config.json"), "w"), indent=2)
    # mgt_report.py compares trained routing against routing at INIT, and the
    # init file is a property of the starting weights, not of which events were
    # evaluated -- so the parent run's copy is the right one to use. Without it
    # mgt_report crashes on a missing file.
    for f in ("best.pt", "routing_init.npz"):
        src = os.path.join(args.run, f)
        dst = os.path.join(outdir, f)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)

    role = rt["role"]
    print(f"\n  {len(np.unique(rt['event_id'])):,} events, "
          f"{len(role):,} real nodes")
    chan = rt["channel_id"]
    for ci, cname in enumerate(args.channel):
        sel = chan == ci
        if not sel.any():
            continue
        n_tau = int((role[sel] == 1).sum())
        n_cand = int(rt["is_cand"][sel].sum())
        print(f"    {cname:<16} {int(sel.sum()):>9,} nodes  "
              f"{n_tau:>8,} tau  {n_cand:>8,} 3pi candidates")
        if n_tau == 0:
            print(f"      note: no tau vertices labelled. Expected for a "
                  f"prompt hadronic mode; for a ...TauNu mode it means "
                  f"vertex_truth.py did not recognise the topology.")
    print(f"\nwrote {outdir}/  (routing_test.npz, preds_test.npz, config.json)")
    print(f"now run:\n  python3 mgt_report.py --run {outdir}\n"
          f"  python3 attention_maps.py --run {outdir} --fig 8")


if __name__ == "__main__":
    main()