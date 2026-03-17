"""
Generate Retrieval Evaluation Charts
=====================================
Usage:
    python evaluation/generate_retrieval_charts.py

Requires:
    evaluation/results/retrieval_summary.json
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('evaluation/charts', exist_ok=True)

with open('evaluation/results/retrieval_summary.json') as f:
    all_metrics = json.load(f)

methods = [m['method'] for m in all_metrics]
COLORS = {'naive': '#888888', 'hyde': '#2196F3', 'hybrid': '#FF9800', 'crag': '#4CAF50'}
colors = [COLORS.get(m, '#333') for m in methods]


# ═══ CHART 1: Overall Retrieval Comparison ═══
fig, ax = plt.subplots(figsize=(10, 6))
metric_keys = ['avg_precision_at_k', 'avg_recall_at_k', 'avg_mrr']
labels = ['Precision@k', 'Recall@k', 'MRR']
x = np.arange(len(labels))
width = 0.18

for i, m in enumerate(all_metrics):
    vals = [m[k] for k in metric_keys]
    bars = ax.bar(x + i * width, vals, width,
                  label=m['method'].upper(), color=colors[i], edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        if val > 0.02:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Retrieval Quality: Naive vs HyDE vs Hybrid vs CRAG', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.15)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('evaluation/charts/1_retrieval_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 1: Overall retrieval comparison saved.')


# ═══ CHART 2: Individual Retrieval Metrics ═══
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ret_metrics = [('avg_precision_at_k', 'Precision@k'),
               ('avg_recall_at_k', 'Recall@k'),
               ('avg_mrr', 'MRR')]

for ax, (key, title) in zip(axes, ret_metrics):
    vals = [m[key] for m in all_metrics]
    bars = ax.bar([m.upper() for m in methods], vals, color=colors, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Retrieval Metrics Breakdown', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('evaluation/charts/2_retrieval_metrics_detail.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 2: Retrieval metrics detail saved.')


# ═══ CHART 3: Performance by Query Type ═══
all_qt = set()
for m in all_metrics:
    all_qt.update(m['by_query_type'].keys())
query_types = sorted(all_qt)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metric_names = [('precision', 'Precision@k'), ('recall', 'Recall@k'), ('mrr', 'MRR')]

for ax, (metric_key, metric_title) in zip(axes, metric_names):
    x = np.arange(len(query_types))
    width = 0.18
    for i, m in enumerate(all_metrics):
        vals = [m['by_query_type'].get(qt, {}).get(metric_key, 0) for qt in query_types]
        ax.bar(x + i * width, vals, width, label=m['method'].upper(), color=colors[i],
               edgecolor='white', linewidth=0.5)
    ax.set_title(metric_title, fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(query_types, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Retrieval Performance by Query Type', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('evaluation/charts/3_by_query_type.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 3: By query type saved.')


# ═══ CHART 4: Performance by Category ═══
all_cats = set()
for m in all_metrics:
    all_cats.update(m['by_category'].keys())
categories = sorted(all_cats)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (metric_key, metric_title) in zip(axes, metric_names):
    x = np.arange(len(categories))
    width = 0.18
    for i, m in enumerate(all_metrics):
        vals = [m['by_category'].get(cat, {}).get(metric_key, 0) for cat in categories]
        ax.bar(x + i * width, vals, width, label=m['method'].upper(), color=colors[i],
               edgecolor='white', linewidth=0.5)
    ax.set_title(metric_title, fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(categories, fontsize=9, rotation=20, ha='right')
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Retrieval Performance by Action Category', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('evaluation/charts/4_by_category.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 4: By category saved.')


# ═══ CHART 5: Improvement Over Baseline ═══
fig, ax = plt.subplots(figsize=(10, 6))
baseline = all_metrics[0]  # naive is first
metric_keys_all = ['avg_precision_at_k', 'avg_recall_at_k', 'avg_mrr']
labels = ['Precision@k', 'Recall@k', 'MRR']
x = np.arange(len(labels))
width = 0.25

for i, m in enumerate(all_metrics[1:], start=0):  # skip naive
    improvements = []
    for key in metric_keys_all:
        base_val = baseline[key]
        new_val = m[key]
        if base_val > 0:
            pct_change = ((new_val - base_val) / base_val) * 100
        else:
            pct_change = 0
        improvements.append(pct_change)

    bars = ax.bar(x + i * width, improvements, width,
                  label=m['method'].upper(), color=COLORS.get(m['method'], '#333'),
                  edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (1 if val >= 0 else -3),
                f'{val:+.1f}%', ha='center', fontsize=9, fontweight='bold')

ax.set_ylabel('% Improvement over Naive Baseline', fontsize=12)
ax.set_title('Retrieval Improvement Over Naive RAG Baseline', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=10)
ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('evaluation/charts/5_improvement_over_baseline.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 5: Improvement over baseline saved.')


print(f'\nAll charts saved to evaluation/charts/')
