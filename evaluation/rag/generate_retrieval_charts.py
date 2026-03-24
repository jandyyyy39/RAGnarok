"""
Generate Retrieval Evaluation Charts
=====================================
Usage:
    python evaluation/rag/generate_retrieval_charts.py

Requires:
    evaluation/rag/results_v2/retrieval_summary.json
"""

import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Make imports work regardless of execution directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, BASE_DIR)

RESULTS_PATH = os.path.join(BASE_DIR, 'evaluation', 'rag', 'results_v2', 'retrieval_summary.json')
CHARTS_DIR = os.path.join(BASE_DIR, 'evaluation', 'rag', 'charts')

os.makedirs(CHARTS_DIR, exist_ok=True)

with open(RESULTS_PATH) as f:
    all_metrics = json.load(f)

# Ensure 'naive' (baseline) is first, then sort alphabetically for consistent chart order
all_metrics.sort(key=lambda x: (x['method'] != 'naive_old', x['method']))


methods = [m['method'] for m in all_metrics]
COLORS = {'naive_old': '#888888', 'hyde': '#2196F3', 'hybrid': '#FF9800', 'crag': '#4CAF50', 'cosine': '#E91E63', 'bm25': '#9C27B0', 'cosine_rerank': '#F44336', 'bm25_rerank': '#673AB7'}
colors = [COLORS.get(m, '#333') for m in methods]
method_labels = [m.replace('_', ' ').upper() for m in methods]


# ═══ CHART 1: Overall Retrieval Comparison ═══
fig, ax = plt.subplots(figsize=(12, 7))
metric_keys = ['avg_precision_at_k', 'avg_recall_at_k', 'map', 'avg_mrr']
labels = ['Precision@k', 'Recall@k', 'mAP', 'MRR']
x = np.arange(len(labels))
width = 0.08

for i, m in enumerate(all_metrics):
    vals = [m[k] for k in metric_keys]
    bars = ax.bar(x + i * width, vals, width,
                  label=method_labels[i], color=colors[i], edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        if val > 0.02:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Overall Retrieval Quality Comparison', fontsize=16, fontweight='bold')
ax.set_xticks(x + width * (len(methods) - 1) / 2)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=10, ncol=2)
ax.set_ylim(0, 1.2)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '1_retrieval_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Chart 1: Overall retrieval comparison saved.')


# ═══ CHART 2: Individual Retrieval Metrics ═══
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
ret_metrics = [('avg_precision_at_k', 'Precision@k'),
               ('avg_recall_at_k', 'Recall@k'),
               ('map', 'Mean Average Precision (mAP)'),
               ('avg_mrr', 'Mean Reciprocal Rank (MRR)')]

for ax, (key, title) in zip(axes.flatten(), ret_metrics):
    vals = [m[key] for m in all_metrics]
    bars = ax.bar(method_labels, vals, color=colors, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 0.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', rotation=25, labelsize=9)


plt.suptitle('Retrieval Metrics Breakdown', fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(CHARTS_DIR, '2_retrieval_metrics_detail.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Chart 2: Retrieval metrics detail saved.')


# # ═══ CHART 3: Performance by Query Type ═══
# all_qt = set()
# for m in all_metrics:
#     if 'by_query_type' in m:
#         all_qt.update(m['by_query_type'].keys())
# query_types = sorted(all_qt)

# if query_types:
#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#     metric_names = [('precision', 'Precision@k'), ('recall', 'Recall@k'), ('mrr', 'MRR')]

#     for ax, (metric_key, metric_title) in zip(axes, metric_names):
#         x = np.arange(len(query_types))
#         width = 0.18
#         for i, m in enumerate(all_metrics):
#             vals = [m.get('by_query_type', {}).get(qt, {}).get(metric_key, 0) for qt in query_types]
#             ax.bar(x + i * width, vals, width, label=method_labels[i], color=colors[i],
#                    edgecolor='white', linewidth=0.5)
#         ax.set_title(metric_title, fontsize=13, fontweight='bold')
#         ax.set_xticks(x + width * 1.5)
#         ax.set_xticklabels(query_types, fontsize=10)
#         ax.set_ylim(0, 1.15)
#         ax.legend(fontsize=8)
#         ax.grid(axis='y', alpha=0.3, linestyle='--')
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)

#     plt.suptitle('Retrieval Performance by Query Type', fontsize=15, fontweight='bold', y=1.02)
#     plt.tight_layout()
#     plt.savefig(os.path.join(CHARTS_DIR, '3_by_query_type.png'), dpi=150, bbox_inches='tight')
#     plt.close()
#     print('Chart 3: By query type saved.')
# else:
#     print("Chart 3: Skipped (no 'by_query_type' data found).")


# # ═══ CHART 4: Performance by Category ═══
# all_cats = set()
# for m in all_metrics:
#     if 'by_category' in m:
#         all_cats.update(m['by_category'].keys())
# categories = sorted(all_cats)

# if categories:
#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#     metric_names = [('precision', 'Precision@k'), ('recall', 'Recall@k'), ('mrr', 'MRR')]

#     for ax, (metric_key, metric_title) in zip(axes, metric_names):
#         x = np.arange(len(categories))
#         width = 0.18
#         for i, m in enumerate(all_metrics):
#             vals = [m.get('by_category', {}).get(cat, {}).get(metric_key, 0) for cat in categories]
#             ax.bar(x + i * width, vals, width, label=method_labels[i], color=colors[i],
#                    edgecolor='white', linewidth=0.5)
#         ax.set_title(metric_title, fontsize=13, fontweight='bold')
#         ax.set_xticks(x + width * 1.5)
#         ax.set_xticklabels(categories, fontsize=9, rotation=20, ha='right')
#         ax.set_ylim(0, 1.15)
#         ax.legend(fontsize=8)
#         ax.grid(axis='y', alpha=0.3, linestyle='--')
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)

#     plt.suptitle('Retrieval Performance by Action Category', fontsize=15, fontweight='bold', y=1.02)
#     plt.tight_layout()
#     plt.savefig(os.path.join(CHARTS_DIR, '4_by_category.png'), dpi=150, bbox_inches='tight')
#     plt.close()
#     print('Chart 4: By category saved.')
# else:
#     print("Chart 4: Skipped (no 'by_category' data found).")


# ═══ CHART 5: Improvement Over Baseline ═══
baseline = next((m for m in all_metrics if m['method'] == 'naive_old'), None)
if baseline:
    fig, ax = plt.subplots(figsize=(12, 7))
    # Use 'map' instead of 'avg_mrr' for main comparison
    metric_keys_all = ['avg_precision_at_k', 'avg_recall_at_k', 'map']
    labels = ['Precision@k', 'Recall@k', 'mAP']
    x = np.arange(len(labels))
    width = 0.15

    non_baseline_metrics = [m for m in all_metrics if m['method'] != 'naive_old']
    
    for i, m in enumerate(non_baseline_metrics):
        improvements = []
        for key in metric_keys_all:
            base_val = baseline.get(key, 0)
            new_val = m.get(key, 0)
            if base_val > 0:
                pct_change = ((new_val - base_val) / base_val) * 100
            else:
                pct_change = 0 if new_val == 0 else float('inf') # Handle division by zero
            improvements.append(pct_change)

        bars = ax.bar(x + i * width, improvements, width,
                      label=m['method'].replace('_', ' ').upper(), color=COLORS.get(m['method'], '#333'),
                      edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, improvements):
            if val != float('inf'):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + (2 if val >= 0 else -5),
                        f'{val:+.1f}%', ha='center', fontsize=9, fontweight='bold')

    ax.set_ylabel('% Improvement over Naive Baseline', fontsize=12)
    ax.set_title('Retrieval Improvement Over Naive RAG Baseline', fontsize=16, fontweight='bold')
    ax.set_xticks(x + width * (len(non_baseline_metrics) - 1) / 2)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=10, ncol=2)
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, '5_improvement_over_baseline.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Chart 5: Improvement over baseline saved.')
else:
    print("Chart 5: Skipped (baseline 'naive_old' method not found).")


print(f'\nAll charts saved to {CHARTS_DIR}')
