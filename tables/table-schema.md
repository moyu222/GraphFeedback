# Table Schema

All report tables are generated or copied from saved experiment outputs; quantitative cells are not manually invented.

## Main performance table

- Source: `experiments/outputs/stress30v1/citeseer/summary.csv`
- Unit: 30 initially correct nodes per method
- Columns: method, successes, ASR with 95% bootstrap interval, mean victim queries, median margin reduction, median semantic similarity, median changed-token ratio

## Feedback-round ablation

- Source: `tables/feedback_round_ablation.csv`
- Generator: `experiments/scripts/analyze_advisor_revision.py`
- Columns: feedback method, matched one-round method, first-round successes, refinement-only successes, total successes

## Per-class success

- Source: `tables/per_class_success.csv`
- Generator: `experiments/scripts/analyze_advisor_revision.py`
- Unit: five selected nodes per CiteSeer class
- Columns: class, sample count, success counts for six methods

## Method diagnostics

- Source: `tables/method_diagnostics.csv`
- Generator: `experiments/scripts/analyze_advisor_revision.py`
- Columns: records, successful nodes, mean valid candidates, zero-valid records, recomputed mean queries, no-valid rate, generation minutes

## Qualitative case

- Source: `tables/qualitative_case.json`
- Generator: `experiments/scripts/analyze_advisor_revision.py`
- Selection rule: refinement-only GraphFeedback success maximising semantic similarity minus changed-token ratio
- Fields: node and class transition, margins, quality metrics, query count, round, and exact saved texts

## Confirmatory random-sample tables

- Sources: `experiments/outputs/random60v1/citeseer/` and `experiments/outputs/stress30v1/citeseer/`
- Generator: `python experiments/scripts/analyze_advisor_revision.py --run-id random60v1`
- Outputs: `random60v1_per_class_success.csv`, `random60v1_method_diagnostics.csv`, `random60v1_feedback_round_ablation.csv`, `random60v1_qualitative_case.json`, and `stress30v1_vs_random60v1.csv`
- Unit: 60 non-overlapping initially correct nodes, 10 per CiteSeer class, for the random run
- Claim rule: keep stress and random outcomes separate; the cross-run table is descriptive and is not a paired test
