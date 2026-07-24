# GraphFeedback Final Local Experiment Report

- Run: `stress30v1`
- Dataset: CiteSeer
- Clean test accuracy: 0.6928 (442/638)
- Attacked sample: 30 initially correct nodes
- Selection strategy: `stratified_low_margin`

## Main results

| Method | ASR (95% bootstrap CI) | Mean queries | Median margin reduction | Median semantic similarity | Generation time (s) |
|---|---:|---:|---:|---:|---:|
| feedback_non_graph | 0.233 [0.100, 0.400] | 2.77 | 0.1037 | 0.987 | 1050.8 |
| generic_paraphrase | 0.167 [0.033, 0.300] | 1.80 | 0.0892 | 0.988 | 735.0 |
| graph_feedback | 0.233 [0.100, 0.400] | 2.50 | 0.1962 | 0.990 | 1042.9 |
| graph_prompt_attack | 0.167 [0.033, 0.300] | 1.90 | 0.1830 | 0.992 | 776.2 |
| non_graph_attack | 0.133 [0.033, 0.267] | 1.80 | 0.0548 | 0.992 | 748.2 |
| random_edit | 0.167 [0.033, 0.300] | 5.63 | 0.0709 | 0.998 | 0.0 |

## Paired comparisons

- graph_prompt_attack vs non_graph_attack: left-only successes=4, right-only successes=3, exact paired p=1.0000.
- graph_feedback vs feedback_non_graph: left-only successes=5, right-only successes=5, exact paired p=1.0000.
- graph_feedback vs graph_prompt_attack: left-only successes=2, right-only successes=0, exact paired p=0.5000.

## Claim boundary

These results describe one released GraphCLIP checkpoint, one CiteSeer sample, one local generator, and fixed budgets. They do not establish broad model insecurity or guaranteed semantic preservation.
