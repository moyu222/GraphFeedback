# GraphFeedback: Score-Guided Textual Adversarial Robustness Evaluation of GraphCLIP

## Abstract

Text-attributed graph models combine node semantics with graph structure, making raw node text a potential perturbation surface even when the topology remains unchanged. This study evaluates whether score feedback can improve prompt-based textual robustness testing of GraphCLIP under black-box access. We introduce GraphFeedback, a two-round search procedure that first queries three candidate texts, selects the candidate with the most favorable class-score change, and returns the observed score and classification-margin changes to a local language model for refinement. The complete procedure is limited to six victim-model queries and enforces constraints on edit ratio, semantic similarity, text length, and the preservation of protected information.

The evaluation uses the released GraphCLIP checkpoint and CiteSeer. A controlled stress set contains 30 initially correct nodes with low clean classification margins, selected in a class-stratified manner. GraphFeedback changes the predictions of 7 nodes, corresponding to an attack success rate (ASR) of 23.3% with a 95% bootstrap confidence interval of [10.0%, 40.0%]. The one-round graph-aware baseline changes 5 predictions, yielding an ASR of 16.7%. The paired outcomes contain two GraphFeedback-only successes and no baseline-only success, although the exact paired test is not significant (p = 0.5). A score-feedback variant without graph context also reaches 23.3% ASR. The evidence therefore supports the narrower conclusion that a second score-guided round produced additional successful cases in this stress sample; it does not establish an independent ASR benefit from neighborhood context. The findings are limited to one checkpoint, one low-margin sample, and one local generator, and should not be interpreted as a general security assessment of GraphCLIP.

Keywords: text-attributed graphs, GraphCLIP, adversarial robustness, black-box evaluation, score feedback

## 1. Introduction

Text-attributed graphs represent entities through both relational structure and natural-language attributes. They arise in citation networks, product graphs, and social platforms, where a node's text may provide information that is unavailable from topology alone. GraphCLIP learns transferable graph representations through graph-summary contrastive pretraining and invariant learning, enabling zero-shot and few-shot prediction across target graphs [1]. Its use of encoded node text, however, also raises a robustness question: can small changes to an existing node's text alter a graph-level representation sufficiently to change the predicted class, even when the graph structure is fixed?

This work studies that question through an offline, controlled robustness simulation. The evaluation does not poison training data, inject nodes, change edges, or interact with an external service. It modifies only an in-memory copy of the target node text and its reconstructed root feature. The assessed setting is black-box score access: the evaluation procedure observes class probabilities but cannot access GraphCLIP parameters, gradients, or training data.

An earlier local experiment showed that one-round graph-aware prompting could change some GraphCLIP predictions, but its ASR was similar to those of generic paraphrasing and random editing. Neighborhood text and class descriptions altered the wording proposed by the generator, yet they did not consistently guide a small language model toward the victim model's decision boundary. This observation motivates a closed-loop alternative. Rather than generating a fixed set of independent candidates, GraphFeedback queries an initial set, records how each candidate changes the clean-class margin, and uses those observations to guide a second generation round.

The study addresses three questions. The first is whether the feedback round creates successful candidates that are absent from the one-round baseline under the same maximum query budget. The second is whether graph context provides an additional benefit after score feedback is introduced. The third is whether changes in ASR are accompanied by changes in query use, perturbation size, semantic similarity, and generation cost. The intended contribution is a reproducible local evaluation of a two-round score-guided search procedure, not a claim of a new universal graph attack or a broad assessment of graph foundation model security.

## 2. Related Work

Prompt-based textual attacks separate candidate generation from victim-model evaluation. PromptAttack organizes the original input, attack objective, and attack guidance into a structured prompt, then applies fidelity filtering to candidate perturbations at character, word, and sentence levels [2]. Query-efficient black-box methods such as Adversarial Boosting Preference further show that the number and allocation of victim queries are central evaluation dimensions rather than incidental implementation details [3]. These studies motivate the use of constrained generation and explicit query accounting, but they do not incorporate graph context or optimize against an external graph foundation model through iterative score feedback.

Research on text-attributed graph robustness has begun to move from continuous feature perturbations toward interpretable text. Intruding with Words studies graph injection attacks in which new text-bearing nodes are added to an existing graph; it also examines neighborhood-conditioned prompts and multi-round correction [4]. ATAG-LLM similarly uses a language model to generate interpretable injected-node attributes under strict black-box assumptions [5]. These injection settings differ from the present evaluation because they introduce new nodes and edges, whereas GraphFeedback edits only the text of an existing node and keeps all graph structure fixed.

Recent work also broadens the evaluation scope. TGRB considers textual, structural, and hybrid perturbations across graph neural networks, robust graph models, and graph-language models [6]. BadGraph uses graph priors and language-model reasoning to jointly modify text and topology across model families [7]. These studies make graph-aware language-model attacks an established direction. Accordingly, this report does not claim that graph-aware prompting or the use of a language model for graph robustness testing is itself novel. Its narrower focus is whether external victim-score feedback improves topology-preserving text evasion against a released GraphCLIP checkpoint under a small, matched query budget.

## 3. Methodology

### 3.1 Problem Formulation and Threat Model

Let a text-attributed graph be denoted by \(G=(V,E,S)\), where \(S=\{s_i\}\) is the collection of node texts. The evaluated model \(f\) receives a local subgraph centered on node \(v_i\), together with encoded node attributes, and returns a vector of class probabilities. The evaluation considers only nodes that GraphCLIP classifies correctly before perturbation. For a selected node, the procedure may replace its original text \(s_i\) with a candidate \(s'_i\), but it cannot modify the edge set, neighboring texts, positional encodings, class prompts, or model parameters. A candidate is successful when

\[
\arg\max_c f_c(G,s'_i) \neq y_i,
\]

where \(y_i\) is the ground-truth class of the initially correct node.

The procedure observes class scores only. It does not use gradients, internal representations, training labels beyond evaluation bookkeeping, or surrogate-model training. All operations are performed locally on public research artifacts. Candidate texts are applied to an in-memory copy of the target subgraph and are never written back to the source dataset or sent to an external system.

### 3.2 Candidate Constraints

Each candidate is checked against the original text before it is queried. The token-change ratio must not exceed 0.20, and the candidate-to-original length ratio must remain within [0.80, 1.20]. Semantic cosine similarity, measured with `sentence-transformers/all-mpnet-base-v2`, must be at least 0.85. The filtering procedure also preserves numbers, citation markers, explicit negations, polarity terms, acronyms, and detected model names. A candidate that fails parsing or any validity check is recorded but is not submitted to GraphCLIP.

These checks define an operational perturbation boundary rather than a guarantee of semantic equivalence. High embedding similarity can miss changes in technical meaning, and lexical preservation rules cannot cover every domain-specific dependency. The automatic constraints are therefore treated as reproducible filters, while semantic preservation remains a limitation of the evaluation.

### 3.3 One-Round Baselines

The non-graph prompt follows the general PromptAttack structure and asks the generator for local phrase replacements intended to change an unknown classifier's prediction without adding a new topic or an explicit class name. The graph-aware prompt adds GraphCLIP's three highest class scores, their class descriptions, and up to five one-hop neighbor texts. Both variants request six independently generated edit plans.

For every valid candidate, the root text is re-encoded and substituted into a fresh copy of the target ego-graph. If one or more candidates change the prediction, the procedure selects the successful candidate with the highest semantic similarity. Otherwise, it selects the valid candidate that minimizes the clean-class classification margin. Random deletion or swapping and conservative generic paraphrasing serve as additional sanity baselines.

### 3.4 Two-Round Score Feedback

GraphFeedback divides the six-query budget into two stages. The first stage reuses the first three evaluated candidates from the corresponding one-round method, ensuring that the feedback method and its baseline begin with identical candidates. If any of these candidates already changes the prediction, the method stops. Otherwise, the candidate with the smallest clean-class margin becomes the current text for refinement.

For the correct class \(y\), the classification margin of candidate \(s'\) is

\[
m(s') = p_y(s') - \max_{c\neq y}p_c(s').
\]

The second-round prompt receives the original top class and score, the original margin, the best current margin, and the margin changes produced by up to three initial trials. It asks the generator to make different local replacements in the current best text while retaining the global constraints relative to the original. The graph-feedback variant retains the one-hop neighborhood context, whereas `feedback_non_graph` removes it. Up to three valid refinement candidates are queried, giving both feedback methods a maximum of six victim queries.

This procedure is a discrete, heuristic search rather than gradient optimization. Its objective is to reduce \(m(s')\) and obtain a prediction change while satisfying the candidate constraints. It does not provide convergence or optimality guarantees. The refinement stage is valuable only if the observed victim scores help the generator identify a more effective edit direction than independent sampling.

## 4. Experimental Setup

### 4.1 Victim Model and Dataset

The victim is the official released GraphCLIP checkpoint, with SHA-256 hash `e93b19a565446c3d62f53eb31fee08570f94c558de216e1aec805d05532721aa`. On the CiteSeer test split, the reproduced clean accuracy is 69.28%, corresponding to 442 correct predictions among 638 test nodes. Node text is re-encoded with `sentence-transformers/all-MiniLM-L6-v2`. On 30 sampled texts, the median cosine similarity between reconstructed and stored node features is 1.0000, confirming that the evaluation pipeline reproduces GraphCLIP's text-input representation closely enough for controlled root-feature replacement.

The final run, `stress30v1`, contains 30 initially correct nodes. Nodes are grouped by the six CiteSeer classes, and five low-clean-margin nodes are selected from each class using seed 88. This construction deliberately concentrates on difficult cases to make prediction changes observable under a local compute budget. It is a stress sample and must not be interpreted as an estimate of ASR over randomly sampled correct CiteSeer nodes.

### 4.2 Generator and Compared Methods

Candidate edits are generated locally with Qwen2.5-0.5B-Instruct at temperature 0.70 and a maximum of 256 new tokens per attempt. The six evaluated methods are `random_edit`, `generic_paraphrase`, `non_graph_attack`, `graph_prompt_attack`, `feedback_non_graph`, and `graph_feedback`. All generated methods use the same lexical and semantic filters and a maximum victim-query budget of six. The feedback variants reuse the first three candidates of their matched one-round baselines and therefore do not receive a more favorable initial candidate set.

### 4.3 Metrics and Statistical Reporting

The primary metric is attack success rate over nodes that are initially classified correctly. The report also includes bootstrap 95% confidence intervals, attacked-subset accuracy, clean-class margin reduction, actual victim queries, the proportion of nodes without a valid candidate, semantic similarity, changed-token ratio, and generation time. Binary paired outcomes are summarized by discordant-success counts and an exact paired test. Given the sample size, p-values are used to describe uncertainty rather than to select or suppress findings.

The pipeline stores generation, filtering, evaluation, and feedback records as append-only JSONL files. Per-query feedback trajectories contain candidate scores, margins, and success indicators. Aggregate CSV files and the report table are derived from these saved outputs. Reusing the same run identifier skips completed records, making the experiment resumable without silently replacing earlier results.

### 4.4 Evaluation Scope

The entire study is an offline adversarial robustness evaluation using a public academic dataset and a public research checkpoint. It changes only local text and subgraph copies. No candidate is submitted to a production service, no source graph is modified, and no real users, unauthorized resources, or deployed systems are involved. The term attack success rate follows standard adversarial machine learning terminology and refers only to prediction changes observed within this controlled simulation.

## 5. Results

### 5.1 Main Results

Table 1. Main robustness-evaluation results on the 30-node CiteSeer stress set.

| Method | Successful Nodes | ASR (95% CI) | Mean Queries | Median Margin Reduction | Median Semantic Similarity | Median Changed-Token Ratio |
|---|---:|---:|---:|---:|---:|---:|
| `random_edit` | 5/30 | 16.7% [3.3%, 30.0%] | 5.63 | 0.0709 | 0.9979 | 0.0271 |
| `generic_paraphrase` | 5/30 | 16.7% [3.3%, 30.0%] | 1.80 | 0.0892 | 0.9875 | 0.0494 |
| `non_graph_attack` | 4/30 | 13.3% [3.3%, 26.7%] | 1.80 | 0.0548 | 0.9924 | 0.0576 |
| `graph_prompt_attack` | 5/30 | 16.7% [3.3%, 30.0%] | 1.90 | 0.1830 | 0.9921 | 0.0252 |
| `feedback_non_graph` | 7/30 | 23.3% [10.0%, 40.0%] | 2.77 | 0.1037 | 0.9873 | 0.0616 |
| `graph_feedback` | 7/30 | 23.3% [10.0%, 40.0%] | 2.50 | 0.1962 | 0.9896 | 0.0269 |

GraphFeedback changes 7 of the 30 predictions, compared with 5 for the one-round graph-aware method. The absolute ASR difference is 6.7 percentage points. Its median clean-class margin reduction is 0.1962, compared with 0.1830 for `graph_prompt_attack`. These observations are consistent with score feedback helping on some cases that remain unresolved after one-round generation. The confidence intervals are wide, however, and the result does not establish a stable performance advantage.

The additional search incurs measurable cost. GraphFeedback uses 2.50 victim queries per node on average, compared with 1.90 for the one-round graph-aware baseline. Their recorded generation times are approximately 1042.9 and 776.2 seconds, respectively. Although both methods have a six-query cap, early success, parsing failures, and candidate rejection reduce actual query counts. GraphFeedback retains a median changed-token ratio of 2.69% and a median semantic similarity of 0.9896, indicating that the additional successes do not depend on large edits under the automatic metrics.

### 5.2 Paired Outcomes and the Contribution of Round Two

The paired comparison between GraphFeedback and `graph_prompt_attack` contains two GraphFeedback-only successes and no baseline-only success. This direction agrees with the ASR difference, but the exact paired result is not significant (p = 0.5). The one-round graph and non-graph methods have four and three method-only successes, respectively, with p = 1.0. The earlier apparent benefit of graph-aware prompting is therefore not stable at this sample size.

The saved trajectories clarify how the feedback stage contributes. Five of the seven GraphFeedback successes occur within the first three queries, while the refinement round adds two successful nodes. For `feedback_non_graph`, four successes occur in round one and three are added by round two. The second round therefore produces genuinely new successful candidates rather than merely reselecting first-round outputs.

GraphFeedback and `feedback_non_graph` nevertheless reach the same final ASR of 23.3%. Their paired outcomes contain five method-only successes on each side, and the exact paired result is p = 1.0. The experiment supports the feasibility of victim-score feedback but does not show that neighborhood context independently improves ASR once feedback is available.

### 5.3 Baselines, Perturbation Quality, and Cost

Random editing also reaches 16.7% ASR, showing that the selected low-margin nodes are intrinsically sensitive to small textual changes. The random baseline uses 5.63 queries per node on average and does not direct its edits using victim scores. GraphFeedback achieves a higher ASR and a larger median margin reduction with fewer actual queries than random editing, but the stress-set design prevents this difference from being generalized to randomly sampled nodes.

The median semantic similarities of graph and non-graph feedback are 0.9896 and 0.9873. GraphFeedback changes a median of 2.69% of tokens, compared with 6.16% for `feedback_non_graph`, and also produces a larger median margin reduction. This pattern may indicate that neighborhood context changes the edit direction, allowing fewer token changes to induce a larger score movement. Because the study includes neither repeated seeds nor human semantic assessment, this interpretation remains a hypothesis rather than a demonstrated mechanism.

## 6. Discussion and Limitations

The main empirical finding is that victim-model scores can serve as an external search signal for a small local language model. A one-round prompt can only infer the decision boundary from class descriptions and, in the graph-aware case, neighborhood text. The feedback round directly reveals whether previous candidates increased or decreased the relevant margin. Even with a 0.5B-parameter generator, the second round adds two graph-feedback successes and three non-graph-feedback successes. This evidence favors a closed-loop generation-and-evaluation procedure over simply requesting more independent prompt variants.

Graph context does not produce an independent ASR gain in the present experiment. GraphFeedback and its non-graph counterpart each change seven predictions, and their method-only success counts are symmetric. CiteSeer neighbors may not always provide cues aligned with the target decision boundary. GraphCLIP has already aggregated structural information internally, so raw neighbor text supplied to the generator may not correspond to the representation directions that influence the victim. The small generator may also have limited capacity to interpret long neighborhood context together with numerical feedback. The experiment cannot distinguish among these explanations.

The evaluation has several external-validity limitations. The sample is intentionally restricted to low-margin nodes, for which higher ASR than on a random population is expected. Only CiteSeer, one GraphCLIP checkpoint, one generator, and one seed are evaluated. The 30-node paired sample provides limited statistical precision. Semantic similarity, token-change ratio, and protected-item rules cannot replace human review, and apparently fluent edits may still alter subtle technical meaning. The five random-edit successes further show that not every label change on this stress set can be attributed to targeted search.

These limitations determine the appropriate claim boundary. The report can state that GraphFeedback adds two successful cases relative to one-round graph-aware prompting on the fixed stress sample. It cannot claim statistically significant superiority, general GraphCLIP vulnerability, transferable performance, or guaranteed semantic preservation. A natural extension would evaluate a larger random sample of initially correct nodes and add a small blinded human assessment, rather than increasing the number of prompt templates.

## 7. Conclusion

This study implements a topology-preserving, black-box textual robustness evaluation for GraphCLIP zero-shot node classification. GraphFeedback feeds observed class scores and margin changes from an initial candidate set back to a local language model, which then generates up to three refinement candidates within the same six-query maximum budget. On a 30-node low-margin CiteSeer stress set, GraphFeedback reaches 23.3% ASR, compared with 16.7% for one-round graph-aware prompting. The refinement round adds two successful graph-feedback cases, but the paired difference is not statistically significant. A non-graph feedback variant reaches the same ASR, indicating that the current evidence is stronger for score feedback itself than for an independent neighborhood-context benefit.

The experiment provides a complete and reproducible chain from threat-model definition and constrained candidate generation to per-query trajectories and paired evaluation. Its findings should be read as a positive signal from a local stress test rather than as a broad security conclusion. Within that boundary, GraphFeedback offers a clearer methodological contribution than the original one-round prompt and exposes both the potential and the uncertainty of score-guided text perturbation under limited local compute.

## References

[1] Y. Zhu, H. Shi, X. Wang, et al., “GraphCLIP: Enhancing Transferability in Graph Foundation Models for Text-Attributed Graphs,” in Proceedings of The Web Conference, 2025. https://openreview.net/forum?id=mjzss9Xg76

[2] X. Xu, K. Kong, N. Liu, et al., “An LLM Can Fool Itself: A Prompt-Based Adversarial Attack,” in Proceedings of the International Conference on Learning Representations, 2024. https://arxiv.org/abs/2310.13345

[3] Z. Yu, Z. Chen, and K. He, “Query-Efficient Textual Adversarial Example Generation for Black-Box Attacks,” in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 556–569, 2024. https://aclanthology.org/2024.naacl-long.31/

[4] R. Lei, Y. Hu, Y. Ren, and Z. Wei, “Intruding with Words: Towards Understanding Graph Injection Attacks at the Text Level,” in Advances in Neural Information Processing Systems, vol. 37, 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/584b78c26916e5947c5b0c4ff8e7c960-Abstract-Conference.html

[5] Y. Lyu, C. Li, X. Zhang, and T. Zhang, “Navigating the Black Box: Leveraging LLMs for Effective Text-Level Graph Injection Attacks,” arXiv:2506.13276, 2025. https://arxiv.org/abs/2506.13276

[6] R. Lei, L. Yi, M. He, et al., “Robustness in Text-Attributed Graph Learning: Insights, Trade-offs, and New Defenses,” arXiv:2510.17185, 2025. https://arxiv.org/abs/2510.17185

[7] Z. Chen, Y. Wang, P. Jiao, et al., “Can LLMs Fool Graph Learning? Exploring Universal Adversarial Attacks on Text-Attributed Graphs,” in Proceedings of The Web Conference, 2026. https://arxiv.org/abs/2603.21155
