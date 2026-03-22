import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('evaluation/charts', exist_ok=True)

with open('evaluation/results/summary.json') as f:
    all_metrics = json.load(f)

methods = [m['method'] for m in all_metrics]
COLORS = {'naive': '#888888', 'hyde': '#2196F3', 'hybrid': '#FF9800', 'crag': '#4CAF50'}
colors = [COLORS.get(m, '#333') for m in methods]


# ═══ CHART 1: Overall Comparison ═══
fig, ax = plt.subplots(figsize=(11, 6))
metric_keys = ['avg_precision_at_k', 'avg_recall_at_k', 'avg_mrr', 'avg_accuracy']
labels = ['Precision@k', 'Recall@k', 'MRR', 'Ruling Accuracy']
x = np.arange(len(labels))
width = 0.18

for i, m in enumerate(all_metrics):
    vals = [m[k] for k in metric_keys]
    bars = ax.bar(x + i * width, vals, width,
                  label=m['method'].upper(), color=colors[i], edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        if val > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('RAG Method Comparison: D&D 5e Rules Retrieval', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=10, loc='lower right')
ax.set_ylim(0, 1.15)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('evaluation/charts/1_overall_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 1 saved.')


# ═══ CHART 2: Retrieval Metrics ═══
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ret_metrics = [('avg_precision_at_k', 'Precision@k'),
               ('avg_recall_at_k', 'Recall@k'),
               ('avg_mrr', 'MRR')]

for ax, (key, title) in zip(axes, ret_metrics):
    vals = [m[key] for m in all_metrics]
    bars = ax.bar(methods, vals, color=colors, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Retrieval Quality Metrics', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('evaluation/charts/2_retrieval_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 2 saved.')


# ═══ CHART 3: Accuracy by Query Type ═══
all_qt = set()
for m in all_metrics:
    all_qt.update(m['by_query_type'].keys())
query_types = sorted(all_qt)

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(query_types))
width = 0.18

for i, m in enumerate(all_metrics):
    vals = [m['by_query_type'].get(qt, {}).get('accuracy', 0) for qt in query_types]
    ax.bar(x + i * width, vals, width, label=m['method'].upper(), color=colors[i],
           edgecolor='white', linewidth=0.5)

ax.set_ylabel('Ruling Accuracy', fontsize=12)
ax.set_title('Accuracy by Query Type', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(query_types, fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.15)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('evaluation/charts/3_accuracy_by_query_type.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 3 saved.')


# ═══ CHART 4: Accuracy by Category ═══
all_cats = set()
for m in all_metrics:
    all_cats.update(m['by_category'].keys())
categories = sorted(all_cats)
short_cats = [c.replace('_', '\n') for c in categories]

fig, axes = plt.subplots(1, len(all_metrics), figsize=(5 * len(all_metrics), 6), sharey=True)
if len(all_metrics) == 1:
    axes = [axes]

for i, m in enumerate(all_metrics):
    vals = [m['by_category'].get(cat, {}).get('accuracy', 0) for cat in categories]
    axes[i].bar(short_cats, vals, color=colors[i], edgecolor='white', linewidth=0.5)
    axes[i].set_title(m['method'].upper(), fontsize=13, fontweight='bold')
    axes[i].set_ylim(0, 1.15)
    axes[i].tick_params(axis='x', labelsize=7, rotation=45)
    axes[i].grid(axis='y', alpha=0.3, linestyle='--')
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)

axes[0].set_ylabel('Ruling Accuracy', fontsize=11)
plt.suptitle('Accuracy by Action Category', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('evaluation/charts/4_accuracy_by_category.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 4 saved.')


# ═══ CHART 5: Accuracy vs Latency ═══
fig, ax = plt.subplots(figsize=(9, 6))
for i, m in enumerate(all_metrics):
    ax.scatter(m['avg_latency'], m['avg_accuracy'], s=250,
               color=colors[i], label=m['method'].upper(),
               zorder=5, edgecolor='white', linewidth=1.5)
    ax.annotate(m['method'].upper(),
                (m['avg_latency'], m['avg_accuracy']),
                textcoords='offset points', xytext=(12, 8),
                fontsize=11, fontweight='bold', color=colors[i])

ax.set_xlabel('Average Latency (seconds)', fontsize=12)
ax.set_ylabel('Ruling Accuracy', fontsize=12)
ax.set_title('Quality vs Speed Tradeoff', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('evaluation/charts/5_latency_vs_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 5 saved.')


# ═══ CHART 6: False Positive Rate ═══
fig, ax = plt.subplots(figsize=(8, 5))
fps = [m['false_positive_rate'] for m in all_metrics]
bars = ax.bar(methods, fps, color=colors, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, fps):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.0%}', ha='center', fontsize=11, fontweight='bold')

ax.set_ylabel('False Positive Rate', fontsize=12)
ax.set_title('False Positive Rate: Wrongly Demanded Rolls', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('evaluation/charts/6_false_positive_rate.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 6 saved.')

print(f'\nAll charts saved to evaluation/charts/')