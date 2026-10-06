# Bc2TauNu
Mixture-of-experts Graph Transformer for B and Bc to tau nu 

---

## Pipeline

```
stage-1 flat ntuples (.root, on EOS)
        │  graph_build.py      one .pt shard per ROOT file, truth labels attached
        ▼
     shards/
        │  feature_stats.py    train/val/test split by FILE + train-only statistics
        ▼
    stats.json
        │  train.py            train, then record expert routing on the test split
        ▼
   runs/<tag>/
        │  routing_analysis.py, mgt_report.py, attention_maps.py,
        │  pv_distance_routing.py, compare_runs.py
        ▼
   numbers + figures

held-out decay channels:  graph_build.py --channel probe:<name>
                          → probe_eval.py / probe_fig8.py on an existing run
```

## Requirements

Python 3.9+  and PyTorch 2.3+.
Other details mentioned in `requirements.txt`.
Input data are the FCCAnalyses stage-1 ntuples of the Bc2TauNu case study:

```
/eos/experiment/fcc/ee/analyses/case-studies/flavour/Bc2TauNu/flatNtuples/spring2021/prod_03/
    Batch_Analysis_stage1/     tight preselection, EVT_MVA1 > 0.6   (default)
    Batch_Training_4stage1/    loose preselection, EVT_MVA1 > -1
```

## How to Run

First run the command to obtain the cvmfs stack:
```
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
```

Set three locations once:

```bash
export SHARDS=/path/to/shards          # training shards
export PROBE_SHARDS=/path/to/probe_shards #shards for probes, not for training
export STATS=/path/to/stats.json
export RUNS=/path/to/runs
```

Keep probe shards in a **separate directory** from training shards.

### 1. Build shards

```bash
python3 graph_build.py --channel all --out $SHARDS
python3 graph_build.py --channel Bc2TauNu --out $SHARDS --max_files 8   # one channel
python3 graph_build.py --channel Bc2TauNu --dry_run --max_events_per_file 2000  # diagnostics only
```

Training channels and labels (`y`): Bc2TauNu (0), Bu2TauNu (1), and background
(2) from Zbb_incl, Zcc_incl and four exclusive B → D 3π modes. Already-built shards are skipped unless `--overwrite` is given.

### 2. Run Split and Statistics

```bash
python3 feature_stats.py --shards $SHARDS --out $STATS
```


### 3. Train the model

```bash
# the main configuration: 8 experts, top-2, 3π-candidate information masked
python3 train.py --stats $STATS --out $RUNS --tag n8_k2_s0_masked \
    --n_experts 8 --k 2 --seed 0 --mask_features cand3pi is_3pi_candidate

# EVT node fed straight to the head instead of through attention
python3 train.py --stats $STATS --out $RUNS --tag n8_k2_s0_masked_evtmlp \
    --n_experts 8 --k 2 --seed 0 --mask_features cand3pi is_3pi_candidate \
    --evt_mode mlp

# quick test to see if the setup is running
python3 train.py --stats $STATS --out $RUNS --tag smoke --epochs 2 \
    --limit_train 20000 --limit_eval 5000
```

Useful flags:

| flag | what it does |
|---|---|
| `--mask_features cand3pi is_3pi_candidate` | removes all 3π-candidate information |
| `--evt_mode node\|mlp` | EVT node in attention and pooling / out of the graph, info added at the end as an MLP |
| `--use_moe 0` | plain feed-forward layers, no experts |
| `--balance_weight` | load-balancing loss weight |
| `--n_experts`, `--k`, `--seed` | number of experts, scan axes |

A run directory contains:

```
config.json        every resolved hyperparameter, plus the stats.json path
metrics.json       per-epoch curves and final test metrics
best.pt            checkpoint at best validation macro-AUC
routing_init.npz   expert routing BEFORE training (the analysis baseline)
routing_test.npz   expert routing after training, test split, one row per real node
preds_test.npz     per-event probabilities, labels, EVT_MVA1
```

Analysis scripts find the data through the absolute paths recorded in
`config.json` and `stats.json`, so moving the shards breaks existing runs.

### 4. Analyse a run

```bash
python3 routing_analysis.py --run $RUNS/n8_k2_s0_masked --figures
python3 mgt_report.py       --run $RUNS/n8_k2_s0_masked
python3 attention_maps.py   --run $RUNS/n8_k2_s0_masked --fig 8      
python3 attention_maps.py   --run $RUNS/n8_k2_s0_masked --fig 7      
python3 pv_distance_routing.py --run $RUNS/n8_k2_s0_masked
python3 compare_runs.py $RUNS/n8_k2_s0_masked $RUNS/n8_k2_s0_masked_evtmlp
```

Figures go to `<run>/report/` unless `--out` is given.

### 5. Held-out decay channels

Some exclusive τ modes (Bd2DTauNu, Bs2DsTauNu, Lb2LcTauNu, ...) are used to test whether routing transfers to decays the model has not seen.

```bash
python3 graph_build.py --channel probe:Bs2DsTauNu --max_files 4 --out $PROBE_SHARDS

# expert plots with the probe as its own group, for one or more runs
python3 probe_fig8.py --runs $RUNS/n8_k2_s0_masked $RUNS/n8_k2_s0_masked_evtmlp \
    --probe_shards $PROBE_SHARDS --channel Bs2DsTauNu --max_events 100000

# charm-lifetime scan: one bar per channel, ordered by charm-hadron lifetime
python3 probe_fig8.py --runs $RUNS/n8_k2_s0_masked --probe_shards $PROBE_SHARDS \
    --channel Lb2LcTauNu Bu2D0TauNu Bs2DsTauNu Bd2DTauNu --role 2 --d2pv_role 2

# writes <run>/probe_<channel>/ in run-directory layout, so every analysis
# script above works on it unchanged
python3 probe_eval.py --run $RUNS/n8_k2_s0_masked --probe_shards $PROBE_SHARDS \
    --channel Bs2DsTauNu
python3 probe_eval.py ... --mask_evt     # same weights, event-level node removed at inference
```

### Data utilities, not needed usually

```bash
python3 inspect_ntuple.py '/eos/.../some_sample/*.root'    # can graph_build.py read it
python3 vertex_truth.py validate <sample_dir>               # truth-matching efficiency and purity
python3 vertex_truth.py yields <sample_dir> [...]           # vertices per (role, parent) cell
```

### Self-tests

```bash
python3 model.py --selftest                  # masking, permutation invariance, gradients, routing capture
python3 dataset.py --stats $STATS --selftest # standardisation, padding, edge features vs a reference loop
python3 probe_fig8.py --selftest             # counting and grouping, no data needed
```

---


| file | role |
|---|---|
| `graph_build.py` | stage-1 ntuples → per-file graph shards (50 node features, truth labels) |
| `vertex_truth.py` | reco-to-MC vertex matching, role and parent-context labels |
| `feature_stats.py` | file-level split and train-only standardisation statistics |
| `dataset.py` | standardisation, node shuffling, on-the-fly edge features, loaders |
| `model.py` | MoE graph transformer: edge-biased attention, noisy top-k MoE |
| `train.py` | training loop, divergence guard, routing capture |
| `probe_eval.py` | run a trained model on held-out channels |
| `routing_analysis.py` | specialisation and invariance statistics |
| `mgt_report.py` | PV consistency, BDT comparison, summary figures |
| `attention_maps.py` | role-indexed attention maps and expert-count figures |
| `pv_distance_routing.py` | expert allocation vs distance from the primary vertex |
| `probe_fig8.py` | expert figures with probe channels as separate groups |
| `compare_runs.py` | multi-run comparison table |
| `feature_table_tex.py` | LaTeX feature table from `stats.json` |
| `inspect_ntuple.py` | check a ROOT file against the required branch list |
| `test_evt_mode.py` |	checks for the three --evt_mode settings |
