# Figure Data Manifest

All figures in `figures/output/` are generated from saved project evidence; no values are simulated.

| Figure | Source | Transformation |
|---|---|---|
| `fig1_graphfeedback_workflow` | `experiments/experiment_protocol.md`; implemented pipeline in `experiments/scripts/pipeline.py` | Conceptual workflow diagram; no quantitative values |
| `fig2_main_results` | `experiments/outputs/stress30v1/citeseer/summary.csv` | ASR converted to percentages; asymmetric error bars use the saved 95% bootstrap CI endpoints |
| `fig3_feedback_analysis` | `summary.csv` and `feedback_trajectories.jsonl` in the same output directory | Round-1 and refinement-only successes are counted by node; the scatter uses saved ASR and mean-query fields |
| `fig4_sampling_comparison` | Separate `summary.csv` files from `stress30v1` and `random60v1` | ASR converted to percentages for five methods common to both runs; success-count labels use the saved attempted-node denominators |
| `fig5_end_to_end_experiment` | `experiments/experiment_protocol.md`, both run manifests, and implemented stages in `experiments/scripts/pipeline.py` | Conceptual end-to-end experiment workflow; clean-correct and sample counts are copied from the saved manifests |
| `fig6_feedback_rounds_both_samples` | `tables/feedback_round_ablation.csv` and `tables/random60v1_feedback_round_ablation.csv` | Stacked counts separate shared first-round successes from refinement-only successes in each formal run |
| `fig7_efficiency_reliability` | Separate `summary.csv` files from both formal runs | Panel (a) links each method's mean-query/ASR point across samples; panel (b) compares saved no-valid-candidate rates |
| `fig8_class_outcomes` | `tables/per_class_success.csv` and `tables/random60v1_per_class_success.csv` | Class-method success counts are divided by the fixed per-class denominator for colour scaling and annotated with exact counts |

Generation command:

```powershell
python figures/generate_report_figures.py
```

Outputs are saved as 450-dpi PNG plus vector PDF and SVG. The two samples are disjoint and must remain separate in captions and claims; the cross-run comparison is descriptive rather than paired.
