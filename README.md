# GraphFeedback

GraphFeedback is a reproducible local research artifact for score-guided, topology-preserving textual robustness evaluation of GraphCLIP. It tests whether a bounded second generation round, informed by black-box class scores and margin changes, can identify additional valid label-changing candidates without increasing the six-query maximum budget.

This repository contains the experiment pipeline, frozen configurations, aggregate evidence, deterministic analysis tables, publication figures, and an academic experimental report. Large model weights, downloaded datasets, raw node text, generated candidates, official university forms, and signatures are intentionally excluded.

## Scope and responsible use

The code is intended for controlled offline robustness research with public academic datasets and released checkpoints. Each candidate modifies only an in-memory copy of the target node's text representation. Graph topology, neighbouring features, class prompts, and victim parameters remain fixed. The pipeline does not access gradients or interact with production systems.

## Formal experiments

The released study contains two separate experiments. `stress30v1` uses 30 initially correct, stratified lower-margin CiteSeer nodes and is interpreted as a mechanism-oriented stress test. `random60v1` is a frozen, non-overlapping, class-balanced random sample of 60 initially correct nodes, with ten nodes per class. Results are not pooled.

| Method | Low-margin stress sample | Class-balanced random sample |
|---|---:|---:|
| Random edit | 5/30 (16.7%) | 0/60 (0.0%) |
| Non-graph prompt | 4/30 (13.3%) | 1/60 (1.7%) |
| Graph-aware prompt | 5/30 (16.7%) | 0/60 (0.0%) |
| Non-graph feedback | 7/30 (23.3%) | 2/60 (3.3%) |
| GraphFeedback | 7/30 (23.3%) | 0/60 (0.0%) |

On the stress sample, GraphFeedback found two refinement-only successes beyond its shared first-round candidates. The paired comparison against the one-round graph-aware prompt contained two GraphFeedback-only successes and no baseline-only success (`p = 0.5`). The frozen random experiment did not reproduce the effect. The supported conclusion is therefore margin-sensitive local feasibility, not broad GraphCLIP insecurity or GraphFeedback superiority.

## Repository layout

```text
GraphFeedback/
├── experiments/
│   ├── scripts/
│   │   ├── pipeline.py
│   │   └── analyze_advisor_revision.py
│   ├── config.local.yaml
│   ├── config.random60.yaml
│   ├── experiment_protocol.md
│   ├── run_local.ps1
│   └── run_random60.sh
├── results/
│   ├── stress30v1/
│   └── random60v1/
├── tables/
├── figures/
│   └── output/
├── reports/
│   └── GraphFeedback_Experimental_Report_Academic.docx
├── REPORT.md
└── CITATION.cff
```

## Installation

The pipeline was validated with Python 3.10, PyTorch 2.4.1 with CUDA 12.1, PyTorch Geometric 2.6.1, and Windows PowerShell 5.1. A CUDA-capable GPU is recommended.

```powershell
git clone https://github.com/moyu222/GraphFeedback.git
cd GraphFeedback

conda create -n graphfeedback python=3.10 -y
conda activate graphfeedback
pip install -r experiments/requirements.txt

git clone https://github.com/ZhuYun97/GraphCLIP.git external/GraphCLIP
```

Place the released GraphCLIP checkpoint and CiteSeer target data according to the upstream GraphCLIP repository before validation.

## Running the experiments

```powershell
# Download the three Hugging Face models
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\run_local.ps1 -Mode Download

# Validate the environment, checkpoint, clean accuracy, and feature reconstruction
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\run_local.ps1 -Mode Validate -RunId validation

# Resume or reproduce the 30-node lower-margin stress experiment
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\run_local.ps1 -Mode Final -RunId stress30v1
```

The confirmatory run is launched from Git Bash because its wrapper performs additional integrity checks:

```bash
bash experiments/run_random60.sh
```

Outputs are written below `experiments/outputs/<run-id>/citeseer/`. Raw JSONL files may contain source node text and generated candidates and are ignored by Git. Review them before sharing.

## Reproducibility record

- Victim: released GraphCLIP checkpoint
- Dataset: CiteSeer
- Clean accuracy: 442/638 (69.28%)
- Stress seed and selection: seed 88; five lower-margin correct nodes per class
- Confirmatory seed and selection: seed 240726; ten random correct nodes per class; zero overlap with the stress sample
- Maximum budget: six victim queries per node and method
- Text constraints: at most 20% changed aligned tokens and sentence-embedding similarity of at least 0.85
- Generator: `Qwen/Qwen2.5-0.5B-Instruct`
- Node encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Semantic filter: `sentence-transformers/all-mpnet-base-v2`
- GraphCLIP checkpoint SHA-256: `e93b19a565446c3d62f53eb31fee08570f94c558de216e1aec805d05532721aa`

Exact settings are stored in the two configuration files and the reviewed experiment protocol. Aggregate results appear under `results/`; all secondary tables and figures are derived deterministically from saved formal records.

## Report and citation

The complete interpretation, equations, figures, limitations, and references are provided in [REPORT.md](REPORT.md) and the formatted [academic Word report](reports/GraphFeedback_Experimental_Report_Academic.docx). The study has not been submitted or published. If this artifact is used, cite the repository metadata in `CITATION.cff` together with the original GraphCLIP paper and implementation.
