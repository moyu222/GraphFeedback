# GraphFeedback Final Local Experiment Report

- Run: `random60v1`
- Dataset: CiteSeer
- Clean test accuracy: 0.6928 (442/638)
- Attacked sample: 60 initially correct nodes
- Selection strategy: `stratified_random`

## Main results

| Method | ASR (95% bootstrap CI) | Mean queries | Median margin reduction | Median semantic similarity | Generation time (s) |
|---|---:|---:|---:|---:|---:|
| feedback_non_graph | 0.033 [0.000, 0.083] | 2.87 | 0.0000 | 0.986 | 1946.9 |
| graph_feedback | 0.000 [0.000, 0.000] | 2.27 | 0.0000 | 0.992 | 1914.3 |
| graph_prompt_attack | 0.000 [0.000, 0.000] | 1.53 | 0.0000 | 0.993 | 1257.6 |
| non_graph_attack | 0.017 [0.000, 0.050] | 2.02 | 0.0000 | 0.989 | 1342.9 |
| random_edit | 0.000 [0.000, 0.000] | 5.73 | 0.0000 | 0.998 | 0.0 |

## Paired comparisons

- graph_prompt_attack vs non_graph_attack: left-only successes=0, right-only successes=1, exact paired p=1.0000.
- graph_feedback vs feedback_non_graph: left-only successes=0, right-only successes=2, exact paired p=0.5000.
- graph_feedback vs graph_prompt_attack: left-only successes=0, right-only successes=0, exact paired p=1.0000.

## Claim boundary

These results describe one released GraphCLIP checkpoint, one CiteSeer sample, one local generator, and fixed budgets. They do not establish broad model insecurity or guaranteed semantic preservation.
