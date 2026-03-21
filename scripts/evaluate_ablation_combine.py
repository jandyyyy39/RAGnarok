"""
evaluate_ablation_combine.py

Combines per-configuration evaluation JSON files produced by `scripts/evaluate.py`
into a single ablation summary:
- one aggregated metrics table
- more visualizations (delta vs baseline, heatmaps, input-type breakdown, scatter)

This is intentionally separated from `evaluate.py` so you can:
1) run configs one-by-one (resume after Groq 429s)
2) re-run only the missing configs later
3) still generate one complete ablation study report at the end
"""

import argparse
import json
import math
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PLOTS_DIR = DATA_DIR / "plots"


def _safe_mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def _normalize_metric_value(metric: str, v: float) -> float:
    """
    Normalize metrics to a roughly 0–1 range for plotting.
    Most metrics in our stored results are already 0–1.
    """
    if v is None:
        return 0.0
    if metric == "llm_composite":
        # Stored values are 1–5 (average of 4 criteria).
        return v / 5.0
    if metric == "response_length":
        # Not used in 0–1 charts.
        return v
    return v


def _guess_default_baseline(experiments: list[str]) -> str:
    # Prefer the original baseline used in the pipeline.
    for e in experiments:
        if e.startswith("Full Pipeline"):
            return e
    return experiments[0] if experiments else ""


def load_all_per_config_results(prefix: str) -> list[dict]:
    """
    Loads all `data/<prefix>_*.json` files and returns the concatenated list.
    """
    files = sorted(DATA_DIR.glob(f"{prefix}_*.json"), key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"No per-config files found: data/{prefix}_*.json")

    all_results: list[dict] = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            part = json.load(f)
        all_results.extend(part)
    return all_results


def compute_aggregates(all_results: list[dict]) -> tuple[list[str], dict]:
    """
    Returns:
      experiments (sorted)
      agg[exp][metric] = list of values
    """
    agg = defaultdict(lambda: defaultdict(list))
    experiments_set = set()

    for r in all_results:
        exp = r.get("experiment", "unknown")
        experiments_set.add(exp)
        scores = r.get("scores", {}) or {}

        for k, v in scores.items():
            if v is None:
                continue
            agg[exp][k].append(v)

        latency = r.get("latency_ms", 0)
        if latency is not None:
            agg[exp]["latency_ms"].append(latency)

    experiments = sorted(experiments_set)
    return experiments, agg


def build_and_save_plots(
    all_results: list[dict],
    experiments: list[str],
    agg: dict,
    baseline: str,
    plot_prefix: str = "final_eval_ablation_plot",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[Plots] Missing matplotlib/numpy. Install them and re-run.")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    def save_plot(stem: str) -> Path:
        return PLOTS_DIR / f"{plot_prefix}_{stem}.png"

    # Metrics selected for the 0–1 style plots.
    candidate_metrics = [
        "rougeL",
        "bertscore",
        "bleu",
        "chrf",
        "rule_coverage",
        "distinct1",
        "distinct2",
        "llm_composite",
    ]
    available_metrics = []
    for m in candidate_metrics:
        if any(m in agg[e] and len(agg[e][m]) > 0 for e in experiments):
            available_metrics.append(m)

    # 1) Grouped bar chart (0–1 metrics)
    short_names = [
        e.replace("Full Pipeline (Groq)", "Full")
         .replace("Local FT (No RAG)", "FT+NoRAG")
         .replace("Local FT (RAG On)", "FT+RAG")
         .replace(" (Groq)", "")
        for e in experiments
    ]
    metrics_for_bars = [m for m in available_metrics if m != "llm_composite"]  # add separately if present
    if "llm_composite" in available_metrics:
        metrics_for_bars.append("llm_composite")

    def mean_val(exp: str, metric: str) -> float:
        if metric not in agg[exp] or not agg[exp][metric]:
            return 0.0
        return _safe_mean(agg[exp][metric])

    # Normalize all except latency and response_length (not included here).
    plot_vals = [[_normalize_metric_value(m, mean_val(e, m)) for e in experiments] for m in metrics_for_bars]

    n_exp = len(experiments)
    x = np.arange(n_exp)
    width = 0.12
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#DA8BC3", "#937860", "#2CA02C", "#FF7F0E", "#1F77B4"]
    for i, metric in enumerate(metrics_for_bars):
        offset = (i - len(metrics_for_bars) / 2 + 0.5) * width
        vals = plot_vals[i]
        label = metric.replace("_", " ").title()
        ax.bar(x + offset, vals, width, label=label, color=colors[i % len(colors)], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (normalized 0–1)", fontsize=11)
    ax.set_title("Ablation Metrics by Configuration (combined)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    plt.tight_layout()
    plt.savefig(save_plot("scores_grouped"), dpi=150)
    plt.close()
    print(f"  Saved -> {save_plot('scores_grouped').name}")

    # 2) Delta vs baseline for key metrics
    if baseline and baseline in experiments:
        key_metrics = [m for m in ["rougeL", "bertscore", "chrf", "distinct2"] if m in available_metrics]
        if key_metrics:
            base_vals = {m: mean_val(baseline, m) for m in key_metrics}
            deltas = {e: [mean_val(e, m) - base_vals[m] for m in key_metrics] for e in experiments if e != baseline}

            fig, ax = plt.subplots(figsize=(10, 5))
            d = np.array(list(deltas.values()))
            # Plot just one metric at a time might be clearer; use grouped bars across metrics.
            y_pos = np.arange(d.shape[0])
            width2 = 0.14
            for mi, m in enumerate(key_metrics):
                ax.bar(y_pos + (mi - len(key_metrics) / 2 + 0.5) * width2, d[:, mi], width2, label=m.replace("_", " ").title(), color=colors[mi % len(colors)], edgecolor="white")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(y_pos)
            delta_exps = [e for e in experiments if e != baseline]
            ax.set_xticklabels([e.replace(" (Groq)", "")[:18] for e in delta_exps], rotation=20, ha="right", fontsize=8)
            ax.set_ylabel("Δ vs Baseline (0–1 metrics)", fontsize=11)
            ax.set_title("Ablation Impact (delta from baseline)", fontsize=13, fontweight="bold")
            ax.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(save_plot("deltas_vs_baseline"), dpi=150)
            plt.close()
            print(f"  Saved -> {save_plot('deltas_vs_baseline').name}")

    # 3) Heatmap (config × metric)
    heat_metrics = [m for m in ["rougeL", "bertscore", "bleu", "chrf", "distinct1"] if m in available_metrics]
    if heat_metrics and len(experiments) > 1:
        matrix = np.array([[ _normalize_metric_value(m, mean_val(e, m)) for m in heat_metrics ] for e in experiments])
        fig, ax = plt.subplots(figsize=(10, max(4, len(experiments) * 0.6)))
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(heat_metrics)))
        ax.set_xticklabels([m.replace("_", " ").title() for m in heat_metrics], fontsize=9)
        ax.set_yticks(range(len(experiments)))
        ax.set_yticklabels(short_names, fontsize=9)
        for i in range(len(experiments)):
            for j in range(len(heat_metrics)):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, label="Score (0–1)")
        ax.set_title("Ablation Heatmap (combined)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_plot("metric_heatmap"), dpi=150)
        plt.close()
        print(f"  Saved -> {save_plot('metric_heatmap').name}")

    # 4) Quality vs Latency scatter (using a simple composite)
    # Composite uses the most stable metrics.
    composite_components = [m for m in ["rougeL", "bertscore", "chrf"] if m in available_metrics]
    if composite_components:
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#DA8BC3", "#937860", "#2CA02C", "#FF7F0E", "#1F77B4"]
        for i, e in enumerate(experiments):
            lat_list = agg[e].get("latency_ms", [])
            avg_lat = _safe_mean(lat_list) if lat_list else 0.0
            comp = _safe_mean([mean_val(e, m) for m in composite_components]) if composite_components else 0.0
            ax.scatter(avg_lat, comp, s=140, color=colors[i % len(colors)], label=short_names[i], zorder=3)
            ax.annotate(short_names[i], (avg_lat, comp), textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xlabel("Average Latency (ms)", fontsize=11)
        ax.set_ylabel("Quality Composite (avg of key metrics)", fontsize=11)
        ax.set_title("Quality vs Latency Trade-off (combined)", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_plot("quality_vs_latency"), dpi=150)
        plt.close()
        print(f"  Saved -> {save_plot('quality_vs_latency').name}")

    # 5) Input-type breakdown (ROUGE-L)
    by_type = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        exp = r.get("experiment", "unknown")
        itype = r.get("input_type", "unknown")
        v = r.get("scores", {}).get("rougeL")
        if v is None:
            continue
        by_type[itype][exp].append(v)

    input_types = sorted(by_type.keys())
    if input_types and len(experiments) > 1:
        # Plot up to 4 types for readability
        types_to_plot = input_types[:4]
        fig, axes = plt.subplots(1, len(types_to_plot), figsize=(5 * len(types_to_plot), 4.5), squeeze=False)
        for idx, itype in enumerate(types_to_plot):
            ax = axes[0, idx]
            vals = []
            for e in experiments:
                vals.append(_safe_mean(by_type[itype].get(e, [])))
            ax.bar(range(len(experiments)), vals, color=["#4C72B0" for _ in experiments], edgecolor="white")
            ax.set_xticks(range(len(experiments)))
            ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
            ax.set_ylim(0, 1)
            ax.set_ylabel("ROUGE-L (mean)")
            ax.set_title(f"Input type: {itype}")
        plt.tight_layout()
        plt.savefig(save_plot("scores_by_input_type"), dpi=150)
        plt.close()
        print(f"  Saved -> {save_plot('scores_by_input_type').name}")

    # 6) Diversity summary (distinct-1/2 box plots)
    if "distinct1" in available_metrics or "distinct2" in available_metrics:
        fig, ax = plt.subplots(figsize=(10, 5))
        # Keep it lightweight: show boxplots for distinct-1 only.
        metric = "distinct1" if "distinct1" in available_metrics else "distinct2"
        data = []
        labels = []
        for e in experiments:
            xs = []
            for r in all_results:
                if r.get("experiment") == e:
                    v = r.get("scores", {}).get(metric)
                    if v is not None:
                        xs.append(v)
            if xs:
                data.append(xs)
                labels.append(e.replace(" (Groq)", ""))
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title("Generation diversity distribution (combined)")
            plt.xticks(rotation=20, ha="right", fontsize=8)
            plt.tight_layout()
            plt.savefig(save_plot("diversity_boxplot"), dpi=150)
            plt.close()
            print(f"  Saved -> {save_plot('diversity_boxplot').name}")

    # 7) LLM judge criteria breakdown (if present)
    llm_metrics = ["narrative_quality", "rules_accuracy", "character_voice", "relevance"]
    present_llm = [m for m in llm_metrics if any(m in agg[e] and len(agg[e][m]) > 0 for e in experiments)]
    if present_llm:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(experiments))
        width = 0.16
        for i, m in enumerate(present_llm):
            vals = [mean_val(e, m) for e in experiments]
            ax.bar(x + (i - len(present_llm)/2 + 0.5) * width, vals, width, label=m.replace("_", " ").title(), edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=15, ha="right", fontsize=8)
        ax.set_ylim(0, 5.2)
        ax.set_ylabel("LLM Judge Score (1–5)")
        ax.set_title("LLM-as-Judge Criteria by Configuration", fontweight="bold")
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(save_plot("llm_judge_breakdown"), dpi=150)
        plt.close()
        print(f"  Saved -> {save_plot('llm_judge_breakdown').name}")

    # 8) Configuration ranking by quality composite
    rank_metric = "llm_composite" if "llm_composite" in available_metrics else "rougeL"
    ranked = sorted(experiments, key=lambda e: mean_val(e, rank_metric), reverse=True)
    vals = [mean_val(e, rank_metric) for e in ranked]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(ranked)), vals, color="#4C72B0", edgecolor="white")
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels([r.replace(" (Groq)", "") for r in ranked], fontsize=9)
    ax.invert_yaxis()
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
    ax.set_xlabel(rank_metric.replace("_", " ").title())
    ax.set_title(f"Configuration Ranking by {rank_metric.replace('_', ' ').title()}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_plot("config_ranking"), dpi=150)
    plt.close()
    print(f"  Saved -> {save_plot('config_ranking').name}")

    # 9) Safety pass rate by configuration
    safety_rate = []
    for e in experiments:
        s = [r.get("safety_pass") for r in all_results if r.get("experiment") == e]
        s = [1.0 if v is True else 0.0 for v in s if isinstance(v, bool)]
        safety_rate.append(_safe_mean(s) if s else 0.0)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(short_names, safety_rate, color="#55A868", edgecolor="white")
    ax.set_ylim(0, 1.05)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set_ylabel("Safety Pass Rate")
    ax.set_title("Safety Pass Rate by Configuration", fontweight="bold")
    plt.xticks(rotation=15, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_plot("safety_pass_rate"), dpi=150)
    plt.close()
    print(f"  Saved -> {save_plot('safety_pass_rate').name}")

    # 10) Expected Groq calls per input (cost proxy)
    groq_calls = []
    for e in experiments:
        c = [r.get("expected_groq_calls") for r in all_results if r.get("experiment") == e and r.get("expected_groq_calls") is not None]
        groq_calls.append(_safe_mean(c) if c else 0.0)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(short_names, groq_calls, color="#DD8452", edgecolor="white")
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_ylabel("Expected Groq Calls / Input")
    ax.set_title("Cost Proxy by Configuration", fontweight="bold")
    plt.xticks(rotation=15, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_plot("cost_proxy_groq_calls"), dpi=150)
    plt.close()
    print(f"  Saved -> {save_plot('cost_proxy_groq_calls').name}")


def print_metrics_table(all_results: list[dict], experiments: list[str], agg: dict, baseline: str) -> None:
    # Prefer a stable column order.
    preferred = [
        "rougeL",
        "bertscore",
        "bleu",
        "chrf",
        "rule_coverage",
        "distinct1",
        "distinct2",
        "latency_ms",
    ]

    # Determine which metrics exist.
    metrics = []
    for m in preferred:
        if m == "latency_ms":
            if any(agg[e].get("latency_ms") for e in experiments):
                metrics.append(m)
            continue
        if any(m in agg[e] and len(agg[e][m]) > 0 for e in experiments):
            metrics.append(m)

    def mean(exp: str, metric: str) -> float:
        if metric == "latency_ms":
            return _safe_mean(agg[exp].get("latency_ms", []))
        return _safe_mean(agg[exp].get(metric, []))

    header = ["Experiment"]
    for m in metrics:
        if m == "latency_ms":
            header.append("Latency(ms)")
        else:
            header.append(m)
    print("\n" + "=" * 120)
    print(" | ".join(h.ljust(16)[:16] for h in header))
    print("=" * 120)
    for e in experiments:
        row = [e[:16]]
        for m in metrics:
            v = mean(e, m)
            if m == "latency_ms":
                row.append(f"{int(v):d}")
            else:
                row.append(f"{v:.3f}")
        if e == baseline:
            row[0] = row[0].ljust(16)[:16] + "*"
        print(" | ".join(x.ljust(16)[:16] for x in row))
    print("=" * 120)
    print(f"*Baseline: {baseline}")


def build_markdown_report(all_results: list[dict], experiments: list[str], baseline: str, prefix: str) -> None:
    """
    Generates a concise markdown report that matches the project requirement:
    - quantitative metrics + latency/cost proxy
    - qualitative analysis with example inputs/outputs
    - compare alternatives + trade-off discussion
    - reflective analysis (what worked/failed + direction for improvement)
    - risks and mitigations
    """
    report_path = PLOTS_DIR.parent / f"ablation_report_{prefix}.md"
    try:
        import numpy as np
    except ImportError:
        print("[Report] Missing numpy; skipping markdown report.")
        return

    # ---- Aggregate meta stats ----
    safety_rates = {}
    expected_calls = {}
    for e in experiments:
        sp = [r.get("safety_pass") for r in all_results if r.get("experiment") == e]
        sp_clean = [1.0 if v is True else 0.0 for v in sp if isinstance(v, bool)]
        safety_rates[e] = _safe_mean(sp_clean)

        calls = [r.get("expected_groq_calls") for r in all_results if r.get("experiment") == e]
        calls_clean = [float(c) for c in calls if c is not None]
        expected_calls[e] = _safe_mean(calls_clean)

    # ---- Quality ordering for qualitative examples ----
    def pick_best_worst(exp: str) -> tuple[dict | None, dict | None]:
        rows = [r for r in all_results if r.get("experiment") == exp]
        if not rows:
            return None, None
        key = "llm_composite" if rows[0].get("scores", {}).get("llm_composite") is not None else "rougeL"
        if key == "rougeL":
            def get_score(r): return (r.get("scores", {}) or {}).get("rougeL") or -1.0
        else:
            def get_score(r): return (r.get("scores", {}) or {}).get("llm_composite") or -1.0
        best = max(rows, key=get_score)
        worst = min(rows, key=get_score)
        return best, worst

    # ---- Compute delta helpers for reflective analysis ----
    def mean_metric(exp: str, metric: str) -> float:
        vals = [r.get("scores", {}).get(metric) for r in all_results if r.get("experiment") == exp]
        vals = [v for v in vals if v is not None]
        return _safe_mean([float(v) for v in vals])

    # Key quantitative signals
    metrics_of_interest = ["rougeL", "bertscore", "rule_coverage", "chrf", "latency_ms", "llm_composite"]

    # ---- Build text ----
    lines: list[str] = []
    lines.append(f"# Ablation Study Report ({prefix})")
    lines.append("")
    lines.append(f"**Baseline:** {baseline}")
    lines.append("")
    lines.append("## 1) Quantitative evaluation")
    lines.append("")
    lines.append("Table summarizes mean metrics across the evaluated inputs (typically 8 for the final run).")
    lines.append("")
    lines.append("| Experiment | ROUGE-L | BERTScore | Rule Coverage | chrF | Latency (ms) | LLM Composite | Safety Pass Rate | Expected Groq Calls |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for e in experiments:
        rl = mean_metric(e, "rougeL")
        bs = mean_metric(e, "bertscore")
        rc = mean_metric(e, "rule_coverage")
        c = mean_metric(e, "chrf")
        lat_vals = [r.get("latency_ms") for r in all_results if r.get("experiment") == e]
        lat_vals = [v for v in lat_vals if v is not None]
        lat = _safe_mean([float(v) for v in lat_vals])
        llm = mean_metric(e, "llm_composite")
        sp = safety_rates.get(e, 0.0)
        calls = expected_calls.get(e, 0.0)
        # Keep formatting concise.
        lines.append(f"| {e} | {rl:.3f} | {bs:.3f} | {rc:.3f} | {c:.3f} | {int(lat)} | {llm:.3f} | {sp:.3f} | {calls:.0f} |")

    lines.append("")
    lines.append("### Cost / latency proxy")
    lines.append("")
    lines.append("True monetary cost requires provider-specific token pricing and token counts. For reproducibility, we use latency plus an *expected Groq call count* derived from which agents are enabled (Rules Arbiter, DM Agent, NPC Consistency, and LLM-as-judge). This gives a stable, comparative proxy for cost.")
    lines.append("")

    lines.append("## 2) Qualitative analysis (example runs)")
    lines.append("")
    for e in experiments:
        best, worst = pick_best_worst(e)
        if not best or not worst:
            continue
        lines.append(f"### {e}")
        lines.append("")
        # Best
        for label, row in [("Best", best), ("Worst", worst)]:
            inp = (row.get("input") or "").replace("\n", " ").strip()
            resp = (row.get("response") or "").replace("\n", " ").strip()
            inp_short = inp[:220] + ("..." if len(inp) > 220 else "")
            resp_short = resp[:520] + ("..." if len(resp) > 520 else "")
            scores = row.get("scores", {}) or {}
            lines.append(f"- **{label} example**: ROUGE-L={scores.get('rougeL', None)} | LLM Composite={scores.get('llm_composite', None)}")
            lines.append(f"  - Input: {inp_short}")
            lines.append(f"  - Output: {resp_short}")
        lines.append("")

    # Reflective analysis: what worked/failed
    lines.append("## 3) Reflective analysis (what worked / what did not)")
    lines.append("")
    base_rl = mean_metric(baseline, "rougeL")
    base_rule = mean_metric(baseline, "rule_coverage")
    base_llm = mean_metric(baseline, "llm_composite")

    lines.append("We compare each ablation to the baseline using changes in ROUGE-L (surface overlap), Rule Coverage (mechanics presence), and the rubric-based LLM composite.")
    lines.append("")
    for e in experiments:
        if e == baseline:
            continue
        d_rl = mean_metric(e, "rougeL") - base_rl
        d_rule = mean_metric(e, "rule_coverage") - base_rule
        d_llm = mean_metric(e, "llm_composite") - base_llm
        # Very short heuristic interpretation
        hint = []
        if d_rule < -0.05:
            hint.append("mechanics grounding appears to drop")
        if d_rl < -0.03:
            hint.append("surface similarity to the reference narrative drops")
        if d_llm < -0.2:
            hint.append("overall rubric score worsens")
        if not hint:
            hint_text = "performance is relatively stable on the main signals"
        else:
            hint_text = "; ".join(hint)
        lines.append(f"- **{e} vs baseline**: ΔROUGE-L={d_rl:+.3f}, ΔRuleCoverage={d_rule:+.3f}, ΔLLMComposite={d_llm:+.3f} → {hint_text}.")

    lines.append("")
    lines.append("### Concrete improvement direction for future work")
    lines.append("")
    lines.append("Based on the observed deltas, a practical next step is to reduce the dependency on a single retrieval signal by improving SRD grounding (e.g., caching retrieved rules per query, using higher `k` plus reranking, and adding guardrails so the DM explicitly references the retrieved mechanics). In parallel, add a deterministic evaluation resume/caching strategy to further reduce Groq cost/rate-limit impact during repeated ablation runs.")

    lines.append("")
    lines.append("## 4) Risks and mitigations")
    lines.append("")
    lines.append("- **API rate limits / token budget exhaustion**: mitigate by running experiments sequentially, resuming from partial JSON outputs, and preferring local configurations when possible.")
    lines.append("- **Telemetry / dependency incompatibilities** (e.g., Chroma telemetry): mitigate by disabling telemetry (see README) and pinning compatible NumPy versions.")
    lines.append("- **Evaluation reliability**: mitigate by using multiple complementary metrics (surface overlap + semantic similarity + mechanics presence + rubric-based LLM judge) and sampling outputs qualitatively.")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Report] Saved -> {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="eval_results", help="Per-config json prefix: data/<prefix>_*.json")
    parser.add_argument("--baseline", type=str, default="", help="Baseline experiment display name (must match stored results).")
    parser.add_argument("--plot-prefix", type=str, default="final_eval_ablation_plot", help="Output filename prefix for all generated plots.")
    args = parser.parse_args()

    all_results = load_all_per_config_results(args.prefix)
    experiments, agg = compute_aggregates(all_results)
    baseline = args.baseline.strip() if args.baseline else _guess_default_baseline(experiments)

    print(f"Loaded {len(all_results)} total evaluations across {len(experiments)} configurations.")
    print(f"Baseline: {baseline}")
    print_metrics_table(all_results, experiments, agg, baseline)

    build_and_save_plots(all_results, experiments, agg, baseline, plot_prefix=args.plot_prefix)
    build_markdown_report(all_results, experiments, baseline, args.prefix)

