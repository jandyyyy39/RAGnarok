import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "evaluation/results_v2/header_eval"
SUMMARY_PATH = os.path.join(RESULTS_DIR, "retrieval_summary.json")
CHARTS_DIR = "evaluation/charts_v2/header_eval"

os.makedirs(CHARTS_DIR, exist_ok=True)

with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
    all_metrics = json.load(f)

methods = [m["method"] for m in all_metrics]

COLORS = {
    "naive": "#888888",
    "naive_old": "#9E9E9E",
    "cosine": "#607D8B",
    "bm25": "#795548",
    "cosine_rerank": "#03A9F4",
    "bm25_rerank": "#9C27B0",
    "hyde": "#2196F3",
    "hybrid": "#FF9800",
    "crag": "#4CAF50",
}

colors = [COLORS.get(m, "#333333") for m in methods]
n_methods = len(methods)
group_width = 0.8
bar_width = group_width / max(n_methods, 1)

def safe_get(d, key, default=0.0):
    val = d.get(key, default)
    return val if isinstance(val, (int, float)) else default

def load_avg_latency(method_name):
    path = os.path.join(RESULTS_DIR, f"retrieval_{method_name}.json")
    if not os.path.exists(path):
        return 0.0

    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    vals = [r.get("latency", 0.0) for r in rows if isinstance(r.get("latency", 0.0), (int, float))]
    return sum(vals) / len(vals) if vals else 0.0

# --- CHART 1: Overall Comparison ---
fig, ax = plt.subplots(figsize=(12, 6))
metric_keys = ["avg_precision_at_k", "avg_recall_at_k", "avg_mrr", "avg_ndcg"]
labels = ["Precision@K", "Recall@K", "MRR", "NDCG"]
x = np.arange(len(labels))

for i, m in enumerate(all_metrics):
    vals = [safe_get(m, k) for k in metric_keys]
    offset = (i - (n_methods - 1) / 2) * bar_width
    bars = ax.bar(
        x + offset,
        vals,
        bar_width,
        label=m["method"].upper(),
        color=colors[i],
        edgecolor="white",
        linewidth=0.5,
    )
    for bar, val in zip(bars, vals):
        if val > 0.03:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )

ax.set_ylabel("Score", fontsize=12)
ax.set_title("Header Match Evaluation: Overall Comparison", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=9, loc="lower right", ncol=2)
ax.set_ylim(0, 1.15)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "1_overall_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Chart 1 saved.")

# --- CHART 2: Core Retrieval Metrics ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ret_metrics = [
    ("avg_f1", "F1"),
    ("avg_hit_at_k", "Hit@K"),
    ("map", "MAP"),
]

for ax, (key, title) in zip(axes, ret_metrics):
    vals = [safe_get(m, key) for m in all_metrics]
    bars = ax.bar(methods, vals, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=25)

plt.suptitle("Header Match Evaluation: Core Retrieval Metrics", fontsize=15, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "2_core_retrieval_metrics.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Chart 2 saved.")

# --- CHART 3: Hit@K by Query Type ---
all_qt = set()
for m in all_metrics:
    all_qt.update(m.get("by_query_type", {}).keys())
query_types = sorted(all_qt)

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(query_types))

for i, m in enumerate(all_metrics):
    vals = [safe_get(m.get("by_query_type", {}).get(qt, {}), "hit") for qt in query_types]
    offset = (i - (n_methods - 1) / 2) * bar_width
    ax.bar(
        x + offset,
        vals,
        bar_width,
        label=m["method"].upper(),
        color=colors[i],
        edgecolor="white",
        linewidth=0.5,
    )

ax.set_ylabel("Hit@K", fontsize=12)
ax.set_title("Header Match Evaluation: Hit@K by Query Type", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(query_types, fontsize=11)
ax.legend(fontsize=9, ncol=2)
ax.set_ylim(0, 1.15)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "3_hit_at_k_by_query_type.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Chart 3 saved.")

# --- CHART 4: Hit@K by Category ---
all_cats = set()
for m in all_metrics:
    all_cats.update(m.get("by_category", {}).keys())
categories = sorted(all_cats)
short_cats = [c.replace("_", "\n") for c in categories]

ncols = min(4, len(all_metrics))
nrows = int(np.ceil(len(all_metrics) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.8 * nrows), sharey=True)
axes = np.array(axes).reshape(-1)

for i, m in enumerate(all_metrics):
    vals = [safe_get(m.get("by_category", {}).get(cat, {}), "hit") for cat in categories]
    axes[i].bar(short_cats, vals, color=colors[i], edgecolor="white", linewidth=0.5)
    axes[i].set_title(m["method"].upper(), fontsize=12, fontweight="bold")
    axes[i].set_ylim(0, 1.15)
    axes[i].tick_params(axis="x", labelsize=8, rotation=45)
    axes[i].grid(axis="y", alpha=0.3, linestyle="--")
    axes[i].spines["top"].set_visible(False)
    axes[i].spines["right"].set_visible(False)

for j in range(len(all_metrics), len(axes)):
    axes[j].axis("off")

axes[0].set_ylabel("Hit@K", fontsize=11)
plt.suptitle("Header Match Evaluation: Hit@K by Category", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "4_hit_at_k_by_category.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Chart 4 saved.")

# --- CHART 5: Quality vs Latency ---
fig, ax = plt.subplots(figsize=(9, 6))
for i, m in enumerate(all_metrics):
    x_val = load_avg_latency(m["method"])
    y_val = safe_get(m, "map")
    ax.scatter(
        x_val,
        y_val,
        s=220,
        color=colors[i],
        label=m["method"].upper(),
        zorder=5,
        edgecolor="white",
        linewidth=1.5,
    )
    ax.annotate(
        m["method"].upper(),
        (x_val, y_val),
        textcoords="offset points",
        xytext=(10, 8),
        fontsize=10,
        fontweight="bold",
        color=colors[i],
    )

ax.set_xlabel("Average Latency (seconds)", fontsize=12)
ax.set_ylabel("MAP", fontsize=12)
ax.set_title("Header Match Evaluation: Quality vs Speed", fontsize=14, fontweight="bold")
ax.grid(alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "5_map_vs_latency.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Chart 5 saved.")

# --- CHART 6: Failure Rate ---
fig, ax = plt.subplots(figsize=(9, 5))
failure_rates = [1.0 - safe_get(m, "avg_hit_at_k") for m in all_metrics]
bars = ax.bar(methods, failure_rates, color=colors, edgecolor="white", linewidth=0.5)

for bar, val in zip(bars, failure_rates):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{val:.0%}",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )

ax.set_ylabel("Failure Rate", fontsize=12)
ax.set_title("Header Match Evaluation: Top-K Failure Rate", fontsize=14, fontweight="bold")
ax.set_ylim(0, 1.1)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="x", rotation=25)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "6_failure_rate.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Chart 6 saved.")

print(f"\nAll charts saved to {CHARTS_DIR}/")