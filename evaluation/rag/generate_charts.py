import json
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = 'evaluation/results_v2'
CHARTS_DIR = 'evaluation/charts_v2'

os.makedirs(CHARTS_DIR, exist_ok=True)

with open(os.path.join(RESULTS_DIR, 'retrieval_summary.json'), 'r', encoding='utf-8') as f:
    all_metrics = json.load(f)

methods = [m['method'] for m in all_metrics]

COLORS = {
    'cosine': '#607D8B',
    'bm25': '#795548',
    'cosine_rerank': '#03A9F4',
    'bm25_rerank': '#9C27B0',
    'hyde': '#2196F3',
    'hybrid': '#FF9800',
    'crag': '#4CAF50',
}

colors = [COLORS.get(m, '#333333') for m in methods]
n_methods = len(methods)
group_width = 0.8
bar_width = group_width / max(n_methods, 1)


def safe_get(d, key, default=0.0):
    val = d.get(key, default)
    return val if isinstance(val, (int, float)) else default

def load_avg_latency(method_name):
    """
    Compute average latency from per-query results file.
    Supports summary method names like 'naive' that may map to retrieval_naive_old.json.
    """
    candidate_names = [method_name]

    if method_name == 'naive':
        candidate_names.append('naive_old')

    for name in candidate_names:
        path = os.path.join(RESULTS_DIR, f'retrieval_{name}.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                rows = json.load(f)
            latencies = [
                row.get('latency', 0.0)
                for row in rows
                if isinstance(row.get('latency', 0.0), (int, float))
            ]
            if latencies:
                return sum(latencies) / len(latencies)

    return 0.0

# --- CHART 1: Overall Comparison ---
fig, ax = plt.subplots(figsize=(12, 6))
metric_keys = ['avg_precision_at_k', 'avg_recall_at_k', 'avg_mrr', 'avg_hit_at_k']
labels = ['Precision@K', 'Recall@K', 'MRR', 'Hit@K']
x = np.arange(len(labels))

for i, m in enumerate(all_metrics):
    vals = [safe_get(m, k) for k in metric_keys]
    offset = (i - (n_methods - 1) / 2) * bar_width
    bars = ax.bar(
        x + offset,
        vals,
        bar_width,
        label=m['method'].upper(),
        color=colors[i],
        edgecolor='white',
        linewidth=0.5
    )
    for bar, val in zip(bars, vals):
        if val > 0.05:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{val:.2f}',
                ha='center',
                va='bottom',
                fontsize=7,
                fontweight='bold'
            )

ax.set_ylabel('Score', fontsize=12)
ax.set_title('RAG Method Comparison: D&D 5e Rules Retrieval', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=9, loc='lower right', ncol=2)
ax.set_ylim(0, 1.15)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '1_overall_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Chart 1 saved.')

# --- CHART 2: Retrieval Metrics ---
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
ret_metrics = [
    ('avg_precision_at_k', 'Precision@K'),
    ('avg_recall_at_k', 'Recall@K'),
    ('avg_mrr', 'MRR'),
    ('avg_ndcg', 'NDCG'),
]

for ax, (key, title) in zip(axes, ret_metrics):
    vals = [safe_get(m, key) for m in all_metrics]
    bars = ax.bar(methods, vals, color=colors, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f'{val:.3f}',
            ha='center',
            fontsize=9,
            fontweight='bold'
        )
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', rotation=25)

plt.suptitle('Retrieval Quality Metrics', fontsize=15, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '2_retrieval_metrics.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Chart 2 saved.')


# --- CHART 3: Hit@K by Query Type ---
all_qt = set()
for m in all_metrics:
    all_qt.update(m.get('by_query_type', {}).keys())
query_types = sorted(all_qt)

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(query_types))

for i, m in enumerate(all_metrics):
    vals = [safe_get(m.get('by_query_type', {}).get(qt, {}), 'hit') for qt in query_types]
    offset = (i - (n_methods - 1) / 2) * bar_width
    ax.bar(
        x + offset,
        vals,
        bar_width,
        label=m['method'].upper(),
        color=colors[i],
        edgecolor='white',
        linewidth=0.5
    )

ax.set_ylabel('Hit@K', fontsize=12)
ax.set_title('Hit@K by Query Type', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(query_types, fontsize=11)
ax.legend(fontsize=9, ncol=2)
ax.set_ylim(0, 1.15)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '3_hit_at_k_by_query_type.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Chart 3 saved.')

# --- CHART 4: Hit@K by Category ---
all_cats = set()
for m in all_metrics:
    all_cats.update(m.get('by_category', {}).keys())
categories = sorted(all_cats)
short_cats = [c.replace('_', '\n') for c in categories]

ncols = min(4, len(all_metrics))
nrows = int(np.ceil(len(all_metrics) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.8 * nrows), sharey=True)
axes = np.array(axes).reshape(-1)

for i, m in enumerate(all_metrics):
    vals = [safe_get(m.get('by_category', {}).get(cat, {}), 'hit') for cat in categories]
    axes[i].bar(short_cats, vals, color=colors[i], edgecolor='white', linewidth=0.5)
    axes[i].set_title(m['method'].upper(), fontsize=12, fontweight='bold')
    axes[i].set_ylim(0, 1.15)
    axes[i].tick_params(axis='x', labelsize=8, rotation=45)
    axes[i].grid(axis='y', alpha=0.3, linestyle='--')
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)

for j in range(len(all_metrics), len(axes)):
    axes[j].axis('off')

axes[0].set_ylabel('Hit@K', fontsize=11)
plt.suptitle('Hit@K by Action Category', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '4_hit_at_k_by_category.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Chart 4 saved.')

# --- CHART 5: MAP vs Latency ---
fig, ax = plt.subplots(figsize=(9, 6))
for i, m in enumerate(all_metrics):
    avg_latency = load_avg_latency(m['method'])
    quality = safe_get(m, 'map')

    ax.scatter(
        avg_latency,
        quality,
        s=250,
        color=colors[i],
        label=m['method'].upper(),
        zorder=5,
        edgecolor='white',
        linewidth=1.5
    )
    ax.annotate(
        m['method'].upper(),
        (avg_latency, quality),
        textcoords='offset points',
        xytext=(12, 8),
        fontsize=10,
        fontweight='bold',
        color=colors[i]
    )

ax.set_xlabel('Average Latency (seconds)', fontsize=12)
ax.set_ylabel('MAP', fontsize=12)
ax.set_title('Quality vs Speed Tradeoff', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '5_map_vs_latency.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Chart 5 saved.')

# --- CHART 6: MAP Comparison ---
fig, ax = plt.subplots(figsize=(9, 5))
map_vals = [safe_get(m, 'map') for m in all_metrics]
bars = ax.bar(methods, map_vals, color=colors, edgecolor='white', linewidth=0.5)

for bar, val in zip(bars, map_vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f'{val:.3f}',
        ha='center',
        fontsize=10,
        fontweight='bold'
    )

ax.set_ylabel('MAP', fontsize=12)
ax.set_title('Mean Average Precision by Method', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', rotation=25)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '6_map_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Chart 6 saved.')

print(f'\nAll charts saved to {CHARTS_DIR}/')