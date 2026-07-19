# GraphFeedback

GraphFeedback is a reproducible local research artifact for score-guided, topology-preserving textual robustness evaluation of GraphCLIP. It evaluates whether a second generation round, informed by black-box class scores and margin changes, produces additional successful candidates under a six-query maximum budget.

This repository accompanies the report [GraphFeedback: Score-Guided Textual Adversarial Robustness Evaluation of GraphCLIP](REPORT.md). It contains the experiment pipeline, fixed configuration, protocol, aggregate results, and validation records. Large model weights, downloaded datasets, raw node texts, generated candidates, and personal project documents are intentionally excluded.

## Scope and responsible use

The code is designed for offline robustness research using public academic datasets and released checkpoints. The reported experiment modifies only an in-memory copy of an existing node's text representation. It does not change source datasets, inject nodes, alter graph edges, access gradients, or interact with production systems.

## Reported experiment

The released `stress30v1` run uses the official GraphCLIP checkpoint and a class-stratified set of 30 initially correct, lower-margin CiteSeer nodes. It is a stress evaluation rather than a population estimate.

| Method | Successful nodes | ASR (95% bootstrap CI) | Mean queries | Median margin reduction |
|---|---:|---:|---:|---:|
| GraphFeedback | 7/30 | 23.3% [10.0%, 40.0%] | 2.50 | 0.1962 |
| Graph prompt attack | 5/30 | 16.7% [3.3%, 30.0%] | 1.90 | 0.1830 |
| Feedback without graph context | 7/30 | 23.3% [10.0%, 40.0%] | 2.77 | 0.1037 |

The paired comparison between GraphFeedback and the one-round graph-aware baseline contains two GraphFeedback-only successes and no baseline-only success, with exact paired `p = 0.5`. The evidence supports the limited claim that the second round added successful cases in this fixed stress sample. It does not establish statistically significant superiority, an independent advantage from graph context, broad GraphCLIP insecurity, or guaranteed semantic preservation.

Machine-readable aggregates are stored in [`results/`](results/). The complete interpretation and limitations are in [`REPORT.md`](REPORT.md).

## Repository layout

```text
GraphFeedback/
├── experiments/
│   ├── scripts/pipeline.py
│   ├── config.local.yaml
│   ├── experiment_protocol.md
│   ├── requirements.txt
│   └── run_local.ps1
├── results/
│   ├── summary.csv
│   ├── paired_comparisons.csv
│   ├── paired_comparisons.json
│   ├── validation.json
│   └── report.md
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
New-Item -ItemType Directory -Path external/GraphCLIP/checkpoints -Force
python -m gdown --id 178RikDLXPy-4eMGDhG5V6RzmlJhp-8fy -O external/GraphCLIP/checkpoints/graphclip_checkpoint_download
```

GraphCLIP provides CiteSeer as its sample target dataset. Follow the upstream [GraphCLIP repository](https://github.com/ZhuYun97/GraphCLIP) if its data layout or download instructions change.

## Running the experiment

The first model download may take several minutes. The downloader uses ordinary files rather than symbolic links, which avoids the common Windows privilege error in the Hugging Face cache.

```powershell
# Download the three Hugging Face models used by the pipeline
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\run_local.ps1 -Mode Download

# Validate imports, checkpoint preparation, clean accuracy, and text reconstruction
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\run_local.ps1 -Mode Validate -RunId validation

# Small end-to-end feedback smoke run
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\run_local.ps1 -Mode FinalSmoke -RunId smoke3

# Full 30-node experiment; rerunning the same ID resumes completed records
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\run_local.ps1 -Mode Final -RunId stress30v1
```

The same PowerShell commands can be launched from Git Bash through `powershell.exe`. If Python is not on `PATH`, set `GRAPHFEEDBACK_PYTHON` to the interpreter in the activated environment before running the script.

Outputs are written under `experiments/outputs/<run-id>/citeseer/`. Raw JSONL outputs may contain source node text and generated candidates, so they are ignored by Git by default. Review them before sharing.

## Reproducibility record

- Dataset: CiteSeer
- Seed: 88
- Selection: five lower-margin, initially correct nodes per class
- Generator: `Qwen/Qwen2.5-0.5B-Instruct`
- Node encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Semantic model: `sentence-transformers/all-mpnet-base-v2`
- Released GraphCLIP checkpoint SHA-256: `e93b19a565446c3d62f53eb31fee08570f94c558de216e1aec805d05532721aa`

Exact configuration values are stored in [`experiments/config.local.yaml`](experiments/config.local.yaml), and the reviewed protocol is in [`experiments/experiment_protocol.md`](experiments/experiment_protocol.md).

## Citation

The project report has not been submitted or published. If this research artifact is used, cite the repository metadata in [`CITATION.cff`](CITATION.cff) and the original GraphCLIP paper and implementation.
