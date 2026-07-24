"""Generate conference-style figures from the saved CiteSeer experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "experiments" / "outputs" / "stress30v1" / "citeseer"
RANDOM_DIR = ROOT / "experiments" / "outputs" / "random60v1" / "citeseer"
TABLE_DIR = ROOT / "tables"
OUT_DIR = ROOT / "figures" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#8A8A8A",
    "light": "#E8EEF3",
    "ink": "#25313C",
}

LABELS = {
    "random_edit": "Random edit",
    "generic_paraphrase": "Generic paraphrase",
    "non_graph_attack": "Non-graph prompt",
    "graph_prompt_attack": "Graph-aware prompt",
    "feedback_non_graph": "Non-graph + feedback",
    "graph_feedback": "GraphFeedback",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#6F7780",
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=450, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def box(ax, x, y, w, h, title, body, color, title_color="white"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.1, edgecolor=color, facecolor="white",
    )
    ax.add_patch(patch)
    head = FancyBboxPatch(
        (x, y + h * 0.67), w, h * 0.33,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0, facecolor=color,
    )
    ax.add_patch(head)
    ax.text(x + w / 2, y + h * 0.835, title, ha="center", va="center",
            color=title_color, fontsize=8.2, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center",
            color=COLORS["ink"], fontsize=7.0, linespacing=1.22)


def arrow(ax, x1, y1, x2, y2, label=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.15, color="#5D6873"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.022, label,
                ha="center", va="bottom", fontsize=7.2, color="#5D6873")


def workflow_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.965, "GraphFeedback: score-guided textual evaluation workflow",
            fontsize=11.5, fontweight="bold", color=COLORS["ink"], va="top")

    y, w, h = 0.62, 0.16, 0.22
    xs = [0.02, 0.215, 0.41, 0.605, 0.80]
    box(ax, xs[0], y, w, h, "1  Input",
        "Target-node text\nFixed local graph", COLORS["blue"])
    box(ax, xs[1], y, w, h, "2  Context",
        "Class scores\nLabels + neighbors", COLORS["sky"], COLORS["ink"])
    box(ax, xs[2], y, w, h, "3  Generate",
        "Round 1\n≤ 3 candidates", COLORS["orange"], COLORS["ink"])
    box(ax, xs[3], y, w, h, "4  Validate",
        "Meaning + budget\nProtected content", COLORS["green"])
    box(ax, xs[4], y, w, h, "5  Query",
        "GraphCLIP scores\nMargin feedback", COLORS["purple"])
    for i in range(4):
        arrow(ax, xs[i] + w, y + h / 2, xs[i + 1] - 0.005, y + h / 2)

    box(ax, 0.80, 0.30, 0.16, 0.19, "6  Decide",
        "Success: select\nElse: refine", COLORS["red"])
    arrow(ax, 0.88, 0.62, 0.88, 0.50)
    box(ax, 0.41, 0.30, 0.30, 0.19, "Feedback refinement",
        "Best margin reduction + numeric scores\nRound 2: ≤ 3 revised candidates",
        COLORS["purple"])
    arrow(ax, 0.80, 0.395, 0.715, 0.395)
    arrow(ax, 0.41, 0.395, 0.365, 0.395)
    arrow(ax, 0.365, 0.395, 0.365, 0.60)

    ax.add_patch(FancyBboxPatch((0.02, 0.055), 0.96, 0.09,
                                boxstyle="round,pad=0.012,rounding_size=0.015",
                                facecolor="#F5F7F9", edgecolor="#B9C2CA", linewidth=0.8))
    ax.text(0.5, 0.10,
            "Boundary: text only  •  fixed topology  •  score-only access  •  no gradients or parameters  •  ≤ 6 victim queries",
            ha="center", va="center", fontsize=7.2, color=COLORS["ink"])
    save(fig, "fig1_graphfeedback_workflow")


def main_results_figure(summary: pd.DataFrame) -> None:
    order = [
        "non_graph_attack", "graph_prompt_attack", "feedback_non_graph",
        "graph_feedback", "generic_paraphrase", "random_edit",
    ]
    d = summary.set_index("method").loc[order].reset_index()
    y = np.arange(len(d))
    values = d["asr"].to_numpy() * 100
    low = (d["asr"] - d["asr_ci_low"]).to_numpy() * 100
    high = (d["asr_ci_high"] - d["asr"]).to_numpy() * 100
    colors = [COLORS["gray"], COLORS["sky"], COLORS["orange"],
              COLORS["blue"], "#B5B5B5", "#D0D0D0"]

    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    bars = ax.barh(y, values, height=0.58, color=colors, edgecolor="white", linewidth=0.6)
    ax.errorbar(values, y, xerr=np.vstack([low, high]), fmt="none", ecolor=COLORS["ink"],
                elinewidth=1.0, capsize=3, capthick=1.0)
    ax.set_yticks(y, [LABELS[m] for m in d["method"]])
    ax.invert_yaxis()
    ax.set_xlim(0, 45)
    ax.set_xticks(np.arange(0, 46, 10))
    ax.set_xlabel("Attack success rate (%)")
    ax.set_title("Textual evaluation success on 30 initially correct CiteSeer nodes",
                 loc="left", fontweight="bold", pad=10)
    ax.xaxis.grid(True, color="#DDE2E6", lw=0.7)
    ax.set_axisbelow(True)
    for bar, v, successes in zip(bars, values, d["successful_nodes"]):
        ax.text(v + 1.0, bar.get_y() + bar.get_height() / 2,
                f"{int(successes)}/30", va="center", ha="left", fontsize=8,
                color=COLORS["ink"], fontweight="bold")
    ax.text(0.995, -0.24,
            "Error bars: 95% bootstrap confidence intervals. No pairwise difference was significant.",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#5D6873")
    save(fig, "fig2_main_results")


def derive_round_contribution() -> pd.DataFrame:
    records = []
    with (DATA_DIR / "feedback_trajectories.jsonl").open(encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    rows = []
    for method in ["feedback_non_graph", "graph_feedback"]:
        subset = df[df["method"] == method]
        first = set(subset[(subset["round"] == 1) & subset["success"]]["node_id"])
        second = set(subset[(subset["round"] == 2) & subset["success"]]["node_id"])
        rows.append(
            {
                "method": method,
                "round_1_successes": len(first),
                "refinement_only_successes": len(second - first),
                "total_successes": len(first | second),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "feedback_round_contribution.csv", index=False)
    return out


def feedback_analysis_figure(summary: pd.DataFrame) -> None:
    rounds = derive_round_contribution()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.35), gridspec_kw={"width_ratios": [0.9, 1.25]})

    x = np.arange(2)
    first = rounds["round_1_successes"].to_numpy()
    refine = rounds["refinement_only_successes"].to_numpy()
    ax1.bar(x, first, width=0.58, color=COLORS["blue"], label="Round 1")
    ax1.bar(x, refine, bottom=first, width=0.58, color=COLORS["orange"], label="Refinement only")
    ax1.set_xticks(x, ["Non-graph\n+ feedback", "GraphFeedback"])
    ax1.set_ylim(0, 8)
    ax1.set_yticks(np.arange(0, 9, 2))
    ax1.set_ylabel("Successful nodes (out of 30)")
    ax1.set_title("(a) Contribution by round", loc="left", fontweight="bold")
    ax1.yaxis.grid(True, color="#DDE2E6", lw=0.7)
    ax1.set_axisbelow(True)
    for i, (a, b) in enumerate(zip(first, refine)):
        ax1.text(i, a / 2, str(a), ha="center", va="center", color="white", fontweight="bold")
        ax1.text(i, a + b / 2, f"+{b}", ha="center", va="center", color=COLORS["ink"], fontweight="bold")
    ax1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)

    methods = list(LABELS)
    d = summary.set_index("method").loc[methods].reset_index()
    point_colors = {
        "graph_feedback": COLORS["blue"], "feedback_non_graph": COLORS["orange"],
        "graph_prompt_attack": COLORS["sky"], "non_graph_attack": COLORS["gray"],
        "generic_paraphrase": "#B5B5B5", "random_edit": "#D0D0D0",
    }
    for _, row in d.iterrows():
        m = row["method"]
        ax2.scatter(row["mean_queries"], row["asr"] * 100,
                    s=68 if "feedback" in m else 48,
                    color=point_colors[m], edgecolor="white", linewidth=0.7, zorder=3)
        label_positions = {
            "graph_feedback": (2.35, 24.6, "center"),
            "feedback_non_graph": (2.95, 21.2, "left"),
            "graph_prompt_attack": (2.18, 17.1, "left"),
            "non_graph_attack": (2.00, 12.7, "left"),
            "generic_paraphrase": (2.08, 18.3, "center"),
            "random_edit": (5.18, 18.15, "right"),
        }
        lx, ly, align = label_positions[m]
        ax2.text(lx, ly, LABELS[m], ha=align, va="center", fontsize=7.0,
                 color=COLORS["ink"])
    ax2.set_xlim(1.4, 6.1)
    ax2.set_ylim(10, 27)
    ax2.set_xlabel("Mean victim-model queries per node")
    ax2.set_ylabel("Attack success rate (%)")
    ax2.set_title("(b) Effectiveness–query trade-off", loc="left", fontweight="bold")
    ax2.grid(True, color="#DDE2E6", lw=0.7)
    ax2.set_axisbelow(True)
    fig.suptitle("Feedback adds successes; graph context shows no observed ASR gain",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=COLORS["ink"])
    fig.subplots_adjust(top=0.82, bottom=0.26, wspace=0.42)
    save(fig, "fig3_feedback_analysis")


def sampling_comparison_figure(
    stress_summary: pd.DataFrame, random_summary: pd.DataFrame
) -> None:
    order = [
        "random_edit",
        "non_graph_attack",
        "graph_prompt_attack",
        "feedback_non_graph",
        "graph_feedback",
    ]
    stress = stress_summary.set_index("method").loc[order]
    random = random_summary.set_index("method").loc[order]
    x = np.arange(len(order))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 3.85))
    stress_bars = ax.bar(
        x - width / 2,
        stress["asr"].to_numpy() * 100,
        width,
        color=COLORS["blue"],
        label="Low-margin stress sample (n=30)",
    )
    random_bars = ax.bar(
        x + width / 2,
        random["asr"].to_numpy() * 100,
        width,
        color=COLORS["orange"],
        label="Stratified-random sample (n=60)",
    )
    stress_low = (stress["asr"] - stress["asr_ci_low"]).to_numpy() * 100
    stress_high = (stress["asr_ci_high"] - stress["asr"]).to_numpy() * 100
    random_low = (random["asr"] - random["asr_ci_low"]).to_numpy() * 100
    random_high = (random["asr_ci_high"] - random["asr"]).to_numpy() * 100
    ax.errorbar(
        x - width / 2,
        stress["asr"].to_numpy() * 100,
        yerr=np.vstack([stress_low, stress_high]),
        fmt="none",
        ecolor=COLORS["ink"],
        elinewidth=0.9,
        capsize=2.5,
        zorder=3,
    )
    ax.errorbar(
        x + width / 2,
        random["asr"].to_numpy() * 100,
        yerr=np.vstack([random_low, random_high]),
        fmt="none",
        ecolor=COLORS["ink"],
        elinewidth=0.9,
        capsize=2.5,
        zorder=3,
    )
    x_labels = [
        "Random edit",
        "Non-graph\nprompt",
        "Graph-aware\nprompt",
        "Non-graph\n+ feedback",
        "GraphFeedback",
    ]
    ax.set_xticks(x, x_labels)
    ax.set_ylim(0, 43)
    ax.set_yticks(np.arange(0, 44, 10))
    ax.set_ylabel("Attack success rate (%)")
    fig.suptitle(
        "Observed success is concentrated in the low-margin stress sample",
        x=0.01,
        y=0.98,
        ha="left",
        fontweight="bold",
        fontsize=10.8,
        color=COLORS["ink"],
    )
    ax.yaxis.grid(True, color="#DDE2E6", lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=2,
    )
    for bars, rows, denominator in (
        (stress_bars, stress, 30),
        (random_bars, random, 60),
    ):
        for bar, (_, row) in zip(bars, rows.iterrows()):
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.7,
                f"{int(row['successful_nodes'])}/{denominator}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                color=COLORS["ink"],
                fontweight="bold",
            )
    ax.text(
        0.995,
        -0.23,
        "Runs are disjoint and shown descriptively; they must not be pooled or treated as a paired comparison.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        color="#5D6873",
    )
    fig.subplots_adjust(top=0.74, bottom=0.27)
    save(fig, "fig4_sampling_comparison")


def compact_flow_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    body,
    color,
    title_color="white",
    linewidth=1.1,
    title_size=7.6,
    body_size=6.5,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.014",
        linewidth=linewidth,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    head_h = h * 0.31
    head = FancyBboxPatch(
        (x, y + h - head_h),
        w,
        head_h,
        boxstyle="round,pad=0.010,rounding_size=0.014",
        linewidth=0,
        facecolor=color,
    )
    ax.add_patch(head)
    ax.text(
        x + w / 2,
        y + h - head_h / 2,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
    )
    ax.text(
        x + w / 2,
        y + (h - head_h) / 2,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=COLORS["ink"],
        linespacing=1.25,
    )


def end_to_end_experiment_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle(
        "End-to-end experimental workflow",
        x=0.04,
        y=0.98,
        ha="left",
        fontsize=11.2,
        fontweight="bold",
        color=COLORS["ink"],
    )

    compact_flow_box(
        ax,
        0.02,
        0.68,
        0.205,
        0.20,
        "1  Validate inputs",
        "GraphCLIP checkpoint\nCiteSeer ego-graphs\n442/638 clean correct",
        COLORS["blue"],
    )
    compact_flow_box(
        ax,
        0.27,
        0.68,
        0.205,
        0.20,
        "2  Freeze samples",
        "stress30v1: 5/class\nrandom60v1: 10/class\nzero node-ID overlap",
        COLORS["sky"],
        title_color=COLORS["ink"],
    )
    compact_flow_box(
        ax,
        0.52,
        0.68,
        0.205,
        0.20,
        "3  Generate edits",
        "Five matched methods\n≤ 6 candidates\nfixed prompts + seed",
        COLORS["orange"],
        title_color=COLORS["ink"],
    )
    compact_flow_box(
        ax,
        0.77,
        0.68,
        0.205,
        0.20,
        "4  Filter",
        "length 0.8–1.2\nedit ratio ≤ 0.20\nsimilarity ≥ 0.85",
        COLORS["green"],
    )

    for start, end in ((0.225, 0.27), (0.475, 0.52), (0.725, 0.77)):
        arrow(ax, start, 0.78, end, 0.78)

    compact_flow_box(
        ax,
        0.70,
        0.34,
        0.20,
        0.20,
        "5  Query victim",
        "Re-encode root text\nkeep topology fixed\nrecord six class scores",
        COLORS["purple"],
    )
    compact_flow_box(
        ax,
        0.395,
        0.34,
        0.20,
        0.20,
        "6  Score feedback",
        "select best margin drop\nround 2 refinement\nstop on valid flip",
        COLORS["orange"],
        title_color=COLORS["ink"],
    )
    compact_flow_box(
        ax,
        0.09,
        0.34,
        0.20,
        0.20,
        "7  Save evidence",
        "JSONL candidates\nquery trajectories\nrun manifests + hashes",
        COLORS["blue"],
    )
    arrow(ax, 0.875, 0.68, 0.80, 0.54)
    arrow(ax, 0.70, 0.44, 0.595, 0.44)
    arrow(ax, 0.395, 0.44, 0.29, 0.44)
    feedback = FancyArrowPatch(
        (0.50, 0.54),
        (0.62, 0.68),
        connectionstyle="arc3,rad=0.20",
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.1,
        color="#66727C",
    )
    ax.add_patch(feedback)
    ax.text(
        0.56,
        0.61,
        "round 2 if needed",
        fontsize=6.3,
        color="#66727C",
        rotation=22,
        ha="center",
    )

    compact_flow_box(
        ax,
        0.20,
        0.06,
        0.60,
        0.17,
        "8  Aggregate and report",
        "ASR + bootstrap CI  •  paired tests  •  queries  •  fidelity  •  class diagnostics\nStress and random samples remain separate; all figures regenerate from saved evidence",
        "#506273",
        body_size=6.3,
    )
    arrow(ax, 0.20, 0.34, 0.34, 0.22)
    ax.text(
        0.5,
        0.01,
        "Frozen boundary: text-only edits  •  score-only access  •  no gradients  •  no topology change  •  no result-driven retuning",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#5D6873",
    )
    save(fig, "fig5_end_to_end_experiment")


def feedback_rounds_both_samples_figure() -> None:
    stress = pd.read_csv(TABLE_DIR / "feedback_round_ablation.csv")
    random = pd.read_csv(TABLE_DIR / "random60v1_feedback_round_ablation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharey=False)
    panels = [
        (axes[0], stress, 30, "(a) Low-margin stress sample"),
        (axes[1], random, 60, "(b) Stratified-random sample"),
    ]
    for ax, data, denominator, title in panels:
        data = data.set_index("feedback_method").loc[
            ["feedback_non_graph", "graph_feedback"]
        ]
        x = np.arange(2)
        first = data["first_round_successes"].to_numpy()
        refine = data["refinement_only_successes"].to_numpy()
        ax.bar(x, first, width=0.58, color=COLORS["blue"], label="Round 1")
        ax.bar(
            x,
            refine,
            bottom=first,
            width=0.58,
            color=COLORS["orange"],
            label="Refinement only",
        )
        ymax = 8 if denominator == 30 else 4
        ax.set_ylim(0, ymax)
        ax.set_yticks(np.arange(0, ymax + 1, 1))
        ax.set_xticks(x, ["Non-graph\n+ feedback", "GraphFeedback"])
        ax.set_ylabel(f"Successful nodes (out of {denominator})")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.yaxis.grid(True, color="#DDE2E6", lw=0.7)
        ax.set_axisbelow(True)
        for i, (a, b) in enumerate(zip(first, refine)):
            total = a + b
            ax.text(
                i,
                total + 0.18,
                f"{int(total)}/{denominator}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color=COLORS["ink"],
            )
            if a > 0:
                ax.text(i, a / 2, str(int(a)), ha="center", va="center", color="white")
            if b > 0:
                ax.text(
                    i,
                    a + b / 2,
                    f"+{int(b)}",
                    ha="center",
                    va="center",
                    color=COLORS["ink"],
                    fontweight="bold",
                )
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle(
        "Feedback adds stress-set successes but does not rescue GraphFeedback on random nodes",
        x=0.01,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.subplots_adjust(top=0.78, bottom=0.23, wspace=0.35)
    save(fig, "fig6_feedback_rounds_both_samples")


def efficiency_reliability_figure(
    stress_summary: pd.DataFrame, random_summary: pd.DataFrame
) -> None:
    common = [
        "random_edit",
        "non_graph_attack",
        "graph_prompt_attack",
        "feedback_non_graph",
        "graph_feedback",
    ]
    stress = stress_summary.set_index("method").loc[common]
    random = random_summary.set_index("method").loc[common]
    colors = {
        "random_edit": "#B7BEC5",
        "non_graph_attack": COLORS["gray"],
        "graph_prompt_attack": COLORS["sky"],
        "feedback_non_graph": COLORS["orange"],
        "graph_feedback": COLORS["blue"],
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.35))
    for method in common:
        ax1.plot(
            [stress.loc[method, "mean_queries"], random.loc[method, "mean_queries"]],
            [stress.loc[method, "asr"] * 100, random.loc[method, "asr"] * 100],
            color=colors[method],
            alpha=0.45,
            linewidth=1.0,
            zorder=1,
        )
        ax1.scatter(
            stress.loc[method, "mean_queries"],
            stress.loc[method, "asr"] * 100,
            marker="o",
            s=48,
            color=colors[method],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax1.scatter(
            random.loc[method, "mean_queries"],
            random.loc[method, "asr"] * 100,
            marker="s",
            s=44,
            color=colors[method],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    label_offsets = {
        "random_edit": (-0.12, 1.2, "right"),
        "non_graph_attack": (-0.05, -1.3, "right"),
        "graph_prompt_attack": (0.05, 1.0, "left"),
        "feedback_non_graph": (0.10, -1.3, "left"),
        "graph_feedback": (-0.10, -1.4, "right"),
    }
    short_labels = {
        "random_edit": "Random edit",
        "non_graph_attack": "Non-graph",
        "graph_prompt_attack": "Graph-aware",
        "feedback_non_graph": "Non-graph + FB",
        "graph_feedback": "GraphFeedback",
    }
    for method in common:
        dx, dy, align = label_offsets[method]
        ax1.text(
            stress.loc[method, "mean_queries"] + dx,
            stress.loc[method, "asr"] * 100 + dy,
            short_labels[method],
            ha=align,
            va="center",
            fontsize=6.5,
            color=colors[method],
            fontweight="bold",
        )
    ax1.scatter([], [], marker="o", color="#808080", label="Stress n=30")
    ax1.scatter([], [], marker="s", color="#808080", label="Random n=60")
    ax1.set_xlabel("Mean victim queries per node")
    ax1.set_ylabel("Attack success rate (%)")
    ax1.set_xlim(1.2, 6.1)
    ax1.set_ylim(-1, 27)
    ax1.set_title("(a) Effectiveness–query trade-off", loc="left", fontweight="bold")
    ax1.grid(True, color="#DDE2E6", lw=0.7)
    ax1.legend(frameon=False, loc="upper right", fontsize=7.5)

    y = np.arange(len(common))
    height = 0.34
    ax2.barh(
        y - height / 2,
        stress["no_valid_candidate_rate"].to_numpy() * 100,
        height,
        color=COLORS["blue"],
        label="Stress n=30",
    )
    ax2.barh(
        y + height / 2,
        random["no_valid_candidate_rate"].to_numpy() * 100,
        height,
        color=COLORS["orange"],
        label="Random n=60",
    )
    ax2.set_yticks(
        y,
        [
            "Random edit",
            "Non-graph prompt",
            "Graph-aware prompt",
            "Non-graph + feedback",
            "GraphFeedback",
        ],
    )
    ax2.invert_yaxis()
    ax2.set_xlabel("No-valid-candidate rate (%)")
    ax2.set_xlim(0, 32)
    ax2.set_title("(b) Candidate-generation reliability", loc="left", fontweight="bold")
    ax2.xaxis.grid(True, color="#DDE2E6", lw=0.7)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, loc="upper left", fontsize=7.5)
    fig.suptitle(
        "Random nodes are harder to flip even when valid candidates are generated",
        x=0.01,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.subplots_adjust(top=0.80, bottom=0.20, wspace=0.42)
    save(fig, "fig7_efficiency_reliability")


def class_outcomes_figure() -> None:
    stress = pd.read_csv(TABLE_DIR / "per_class_success.csv")
    random = pd.read_csv(TABLE_DIR / "random60v1_per_class_success.csv")
    methods = [
        ("non_graph_attack_successes", "Non-graph\nprompt"),
        ("graph_prompt_attack_successes", "Graph-aware\nprompt"),
        ("feedback_non_graph_successes", "Non-graph\n+ feedback"),
        ("graph_feedback_successes", "GraphFeedback"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.65), sharey=True)
    for ax, data, denominator, title in (
        (axes[0], stress, 5, "(a) Stress sample"),
        (axes[1], random, 10, "(b) Random sample"),
    ):
        counts = data[[item[0] for item in methods]].to_numpy()
        rates = counts / denominator * 100
        image = ax.imshow(rates, cmap="Blues", vmin=0, vmax=60, aspect="auto")
        ax.set_xticks(
            np.arange(len(methods)),
            [
                "Non-graph\nprompt",
                "Graph-aware\nprompt",
                "Non-graph\nfeedback",
                "Graph\nfeedback",
            ],
        )
        ax.set_yticks(np.arange(len(data)), data["class_name"])
        ax.set_title(title, loc="left", fontweight="bold")
        ax.tick_params(axis="x", rotation=0, labelsize=6.8)
        for row in range(counts.shape[0]):
            for col in range(counts.shape[1]):
                value = int(counts[row, col])
                ax.text(
                    col,
                    row,
                    f"{value}/{denominator}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if rates[row, col] >= 35 else COLORS["ink"],
                    fontweight="bold" if value else "normal",
                )
    cbar = fig.colorbar(image, ax=axes, fraction=0.022, pad=0.040)
    cbar.set_label("Within-class success rate (%)")
    fig.suptitle(
        "Class-level outcomes are sparse and sample-dependent",
        x=0.01,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.subplots_adjust(top=0.82, bottom=0.24, left=0.18, right=0.87, wspace=0.16)
    save(fig, "fig8_class_outcomes")


def main() -> None:
    style()
    summary = pd.read_csv(DATA_DIR / "summary.csv")
    random_summary = pd.read_csv(RANDOM_DIR / "summary.csv")
    workflow_figure()
    main_results_figure(summary)
    feedback_analysis_figure(summary)
    sampling_comparison_figure(summary, random_summary)
    end_to_end_experiment_figure()
    feedback_rounds_both_samples_figure()
    efficiency_reliability_figure(summary, random_summary)
    class_outcomes_figure()
    print(f"Generated figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
