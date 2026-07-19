# Final Local Protocol: GraphFeedback

## Status and purpose

- Deliverable: the final reproducible experiment for a research-style technical report.
- Victim: released frozen GraphCLIP checkpoint.
- Dataset: CiteSeer.
- Main sample: the existing `stress30v1` set of 30 stratified low-margin test nodes that GraphCLIP initially classifies correctly, seed 88.
- Runtime target: reuse the completed base generations and finish the feedback extension locally in roughly 30--90 minutes; all stages are resumable.
- This is a deliberately difficult-node stress evaluation, not a population estimate over random CiteSeer test nodes.

## Research question

Under the same six-query and text-edit budget, does a second generation round informed by actual GraphCLIP score changes improve attack success or query efficiency over independent one-shot candidates, and does graph context add value beyond score feedback alone?

## Input and preprocessing

GraphCLIP reads 384-dimensional MiniLM features from `processed_data/citeseer.pt`. Each candidate text is encoded with `sentence-transformers/all-MiniLM-L6-v2` and replaces only the attacked root feature in a fresh copy of its ego-graph. Topology, neighbor text, positional encodings, class prompts, and victim parameters remain fixed.

## Methods

All methods attack identical selected nodes.

1. `random_edit`: deletion/swap sanity baseline, six candidates.
2. `generic_paraphrase`: conservative meaning-preserving rewrite without an adversarial objective.
3. `non_graph_attack`: six independently prompted PromptAttack-style candidates without graph context.
4. `graph_prompt_attack`: six independently prompted candidates with class-score and one-hop context.
5. `feedback_non_graph`: reuse the first three evaluated `non_graph_attack` candidates as round 1; if they fail, feed their actual score-margin changes back to the generator and query three refinements in round 2.
6. `graph_feedback`: the same two-round procedure initialized from the first three `graph_prompt_attack` candidates and retaining graph context in the refinement prompt.

Reusing the first three baseline candidates makes round 1 identical rather than merely similar and keeps the total victim budget at six. A feedback method stops after round 1 when a valid candidate already flips the label.

## Feedback signal

The generator receives:

- the original top-class name and its original score;
- the current best candidate text;
- the current top-class score and clean-class margin;
- the change in margin relative to the original;
- a compact list of margin changes from up to three first-round candidates.

It does not receive victim parameters, gradients, or the ground-truth label. The second round must edit the current best candidate locally while all validity constraints are checked against the original text.

## Candidate validity

- English text with no explanation wrapper.
- Length ratio in `[0.80, 1.20]` relative to the original.
- Changed-token ratio at most `0.20` relative to the original.
- Semantic cosine similarity at least `0.85` using `all-mpnet-base-v2`.
- Preserve numbers, citation markers, explicit negations, polarity terms, acronyms, and detected model names.
- Invalid or unparsable candidates consume generation attempts but are not sent to the victim.

## Metrics

- Attack success rate with paired bootstrap 95% confidence interval.
- Attacked subset accuracy.
- Clean-class margin reduction.
- Victim queries used and queries per successful attack.
- No-valid-candidate rate.
- Median semantic similarity and changed-token ratio.
- Generation time and failure records.

The main comparisons are `graph_feedback` versus `graph_prompt_attack`, and `graph_feedback` versus `feedback_non_graph`. Exact paired tests are descriptive because the sample is small.

## Execution gates

1. E0 validation must reproduce clean inference and feature reconstruction.
2. A 3-node final-pipeline smoke test must complete all five methods and write feedback trajectories.
3. The final run resumes `stress30v1`; completed base records are immutable and only feedback records are appended.
4. If parsing or valid-record rate falls below 50%, stop and inspect instead of silently changing thresholds.
5. Do not tune on the final sample after reading its aggregate results. Code defects may be fixed, but the change and rerun must be recorded.

## Saved outputs

- `run_manifest.json`, `selection.jsonl`.
- `generations.jsonl`, `filtered.jsonl`, `evaluated.jsonl` for base methods.
- `feedback_generations.jsonl`, `feedback_filtered.jsonl`, `feedback_evaluated.jsonl` for the refinement round.
- `feedback_trajectories.jsonl` with per-query scores and margins.
- `summary.csv`, `paired_comparisons.csv`, and mechanically generated `report.md`.

## Allowed claims

- Report whether feedback improved this GraphCLIP/CiteSeer run under the frozen budget.
- Describe query, semantic, edit, runtime, and failure trade-offs.
- Describe the final result as a controlled low-margin stress evaluation.

## Claims not allowed

- General insecurity of GraphCLIP or graph foundation models.
- Broad superiority across datasets, victims, or generators.
- Guaranteed semantic preservation from an embedding threshold.
- First graph-aware LLM attack or first text-attributed graph attack.
