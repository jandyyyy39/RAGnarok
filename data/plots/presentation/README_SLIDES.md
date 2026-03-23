# Presentation figures (auto-generated)

## SECTION: Hyperparameters (use these PNGs)

| Slide | Suggested file | What it shows |
|-------|----------------|---------------|
| HP-1 Grid (3 axes) | *Mermaid in `docs/presentation_hp_ab_slides.md`* — or `01_hparam_grid_search_space.png` as numeric grid | All (r × α × lr) cells; green box = best |
| HP-2 Efficient search | `03_hparam_ranked_by_loss.png` or `04_hparam_ranked_by_perplexity.png` | All configs sorted; blue = best |
| HP-3 Best → full FT | Same as HP-2 + say verbally: full `finetune_fireball.py` | — |

**Tip:** Show **perplexity** grid (`02_...`) for intuition; **loss** for training purists — same ranking.

## SECTION: Ablation study

| Slide | Suggested file | What it shows |
|-------|----------------|---------------|
| AB-1 Architecture | *Mermaid diagram* in docs (toggles) | — |
| AB-2 Seven configs | *Table on slide* (see READMEeval) | — |
| AB-3 Metrics | Icons + bullets (no single plot required) | — |
| AB-4 Results | `pres_ablation_deltas_vs_baseline.png` or `pres_ablation_scores_grouped.png` | Δ vs baseline or grouped metrics |
| AB-5 Takeaway | `pres_ablation_quality_vs_latency.png` | Quality vs latency trade-off |
| Appendix | `pres_ablation_metric_heatmap.png` | Config × metric heatmap |

Files use prefix `pres_ablation_*` from `--ablation-prefix eval_results8`.

Regenerate everything:
  python scripts/generate_presentation_plots.py
