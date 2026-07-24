"""Generate deterministic advisor-revision tables from a completed run.

This script performs secondary analysis only. It does not modify the saved
experimental records or query the victim model.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "tables"

METHOD_ORDER = [
    "random_edit",
    "generic_paraphrase",
    "non_graph_attack",
    "graph_prompt_attack",
    "feedback_non_graph",
    "graph_feedback",
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def class_index_to_name(selection: list[dict]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for record in selection:
        for item in record["top_classes"]:
            score = float(item["score"])
            index = min(
                range(len(record["clean_scores"])),
                key=lambda candidate: abs(float(record["clean_scores"][candidate]) - score),
            )
            mapping.setdefault(index, item["name"])
    return mapping


def margin(scores: list[float], label: int) -> float:
    alternatives = [float(value) for index, value in enumerate(scores) if index != label]
    return float(scores[label]) - max(alternatives)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="stress30v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = ROOT / "experiments" / "outputs" / args.run_id / "citeseer"
    prefix = "" if args.run_id == "stress30v1" else f"{args.run_id}_"
    selection = read_jsonl(run_dir / "selection.jsonl")
    standard = read_jsonl(run_dir / "evaluated.jsonl")
    feedback = read_jsonl(run_dir / "feedback_evaluated.jsonl")
    all_records = standard + feedback
    selection_by_node = {int(record["node_id"]): record for record in selection}
    class_names = class_index_to_name(selection)

    records_by_node_method: dict[tuple[int, str], dict] = {}
    for record in all_records:
        # Resumable runs may append a completed record again. The canonical
        # aggregation keeps the last record for each node-method pair.
        records_by_node_method[(int(record["node_id"]), record["method"])] = record
    records_by_method: dict[str, list[dict]] = defaultdict(list)
    for record in records_by_node_method.values():
        records_by_method[record["method"]].append(record)
    method_order = [method for method in METHOD_ORDER if method in records_by_method]

    per_class_rows: list[dict] = []
    for label in sorted({int(record["label"]) for record in selection}):
        node_ids = [int(record["node_id"]) for record in selection if int(record["label"]) == label]
        row = {
            "label": label,
            "class_name": class_names.get(label, f"class_{label}"),
            "nodes": len(node_ids),
        }
        for method in method_order:
            row[f"{method}_successes"] = sum(
                bool(records_by_node_method[(node_id, method)]["success"]) for node_id in node_ids
            )
        per_class_rows.append(row)
    per_class_fields = ["label", "class_name", "nodes"] + [
        f"{method}_successes" for method in method_order
    ]
    write_csv(TABLE_DIR / f"{prefix}per_class_success.csv", per_class_rows, per_class_fields)

    with (run_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary_by_method = {row["method"]: row for row in csv.DictReader(handle)}

    diagnostic_rows: list[dict] = []
    for method in method_order:
        records = records_by_method[method]
        summary = summary_by_method[method]
        valid_counts = [int(record["valid_candidate_count"]) for record in records]
        query_counts = [
            int(record.get("query_count", len(record.get("evaluated_candidates", []))))
            for record in records
        ]
        diagnostic_rows.append(
            {
                "method": method,
                "records": len(records),
                "successful_nodes": sum(bool(record["success"]) for record in records),
                "mean_valid_candidates": f"{mean(valid_counts):.2f}",
                "zero_valid_records": sum(count == 0 for count in valid_counts),
                "mean_victim_queries_recomputed": f"{mean(query_counts):.2f}",
                "no_valid_candidate_rate": f"{float(summary['no_valid_candidate_rate']):.4f}",
                "generation_minutes": f"{float(summary['generation_seconds']) / 60:.2f}",
            }
        )
    diagnostic_fields = [
        "method",
        "records",
        "successful_nodes",
        "mean_valid_candidates",
        "zero_valid_records",
        "mean_victim_queries_recomputed",
        "no_valid_candidate_rate",
        "generation_minutes",
    ]
    write_csv(TABLE_DIR / f"{prefix}method_diagnostics.csv", diagnostic_rows, diagnostic_fields)

    round_rows: list[dict] = []
    for method, baseline in [
        ("feedback_non_graph", "non_graph_attack"),
        ("graph_feedback", "graph_prompt_attack"),
    ]:
        records = records_by_method[method]
        first_round = sum(
            any(bool(candidate["success"]) for candidate in record["initial_candidates"])
            for record in records
        )
        refinement_only = sum(
            bool(record["success"])
            and not any(bool(candidate["success"]) for candidate in record["initial_candidates"])
            for record in records
        )
        round_rows.append(
            {
                "feedback_method": method,
                "matched_one_round_method": baseline,
                "first_round_successes": first_round,
                "refinement_only_successes": refinement_only,
                "total_successes": first_round + refinement_only,
                "attempted_nodes": len(records),
            }
        )
    write_csv(
        TABLE_DIR / f"{prefix}feedback_round_ablation.csv",
        round_rows,
        [
            "feedback_method",
            "matched_one_round_method",
            "first_round_successes",
            "refinement_only_successes",
            "total_successes",
            "attempted_nodes",
        ],
    )

    qualitative_candidates: list[tuple[int, float, dict]] = []
    # Prefer a refinement-only GraphFeedback success. If none exists, retain
    # the same deterministic rule for the matched non-graph feedback method.
    for method_priority, method in enumerate(("graph_feedback", "feedback_non_graph")):
        for record in records_by_method[method]:
            initial_success = any(
                bool(candidate["success"]) for candidate in record["initial_candidates"]
            )
            if not bool(record["success"]) or initial_success or not record.get("selected"):
                continue
            selected = record["selected"]
            quality_score = float(selected["semantic_similarity"]) - float(
                selected["changed_token_ratio"]
            )
            qualitative_candidates.append((-method_priority, quality_score, record))
    if not qualitative_candidates:
        raise RuntimeError("No refinement-only feedback success was found")
    _, _, example = max(qualitative_candidates, key=lambda item: (item[0], item[1]))
    selected = example["selected"]
    node_id = int(example["node_id"])
    label = int(example["label"])
    clean_scores = [float(value) for value in example["clean_scores"]]
    attacked_scores = [float(value) for value in selected["scores"]]
    case = {
        "run_id": args.run_id,
        "method": example["method"],
        "node_id": node_id,
        "true_and_clean_class_index": label,
        "true_and_clean_class_name": class_names.get(label, f"class_{label}"),
        "attacked_prediction_index": int(selected["prediction"]),
        "attacked_prediction_name": class_names.get(
            int(selected["prediction"]), f"class_{int(selected['prediction'])}"
        ),
        "clean_margin": margin(clean_scores, label),
        "attacked_margin": margin(attacked_scores, label),
        "margin_reduction": margin(clean_scores, label) - margin(attacked_scores, label),
        "semantic_similarity": float(selected["semantic_similarity"]),
        "changed_token_ratio": float(selected["changed_token_ratio"]),
        "length_ratio": float(selected["length_ratio"]),
        "victim_queries": int(example["query_count"]),
        "success_round": 2,
        "original_text": selection_by_node[node_id]["original_text"],
        "selected_text": selected["text"],
    }
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / f"{prefix}qualitative_case.json").write_text(
        json.dumps(case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if args.run_id == "random60v1":
        stress_summary_path = (
            ROOT / "experiments" / "outputs" / "stress30v1" / "citeseer" / "summary.csv"
        )
        with stress_summary_path.open(encoding="utf-8", newline="") as handle:
            stress_summary = {row["method"]: row for row in csv.DictReader(handle)}
        comparison_rows = []
        for method in method_order:
            if method not in stress_summary:
                continue
            stress = stress_summary[method]
            random = summary_by_method[method]
            comparison_rows.append(
                {
                    "method": method,
                    "stress_successes": int(stress["successful_nodes"]),
                    "stress_attempted": int(stress["attempted_nodes"]),
                    "stress_asr": f"{float(stress['asr']):.4f}",
                    "random_successes": int(random["successful_nodes"]),
                    "random_attempted": int(random["attempted_nodes"]),
                    "random_asr": f"{float(random['asr']):.4f}",
                    "random_minus_stress_asr": (
                        f"{float(random['asr']) - float(stress['asr']):.4f}"
                    ),
                }
            )
        write_csv(
            TABLE_DIR / "stress30v1_vs_random60v1.csv",
            comparison_rows,
            [
                "method",
                "stress_successes",
                "stress_attempted",
                "stress_asr",
                "random_successes",
                "random_attempted",
                "random_asr",
                "random_minus_stress_asr",
            ],
        )

    print(f"Wrote {args.run_id} advisor-revision analysis to {TABLE_DIR}")


if __name__ == "__main__":
    main()
