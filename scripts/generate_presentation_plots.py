"""
Generate a clean set of figures for CS 614 / RAGnarok presentation slides.

Writes to: data/plots/presentation/
  - Hyperparameter figures (from hparam_results.json via plot_hparam_results.py)
  - Ablation figures (from eval per-config JSONs via evaluate_ablation_combine.py)
  - README_SLIDES.md — which file to use on which slide

Usage:
  python scripts/generate_presentation_plots.py
  python scripts/generate_presentation_plots.py --hparam-json data/hparam_results.json
  python scripts/generate_presentation_plots.py --ablation-prefix eval_results8 --skip-ablation
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRESENTATION_DIR = ROOT / "data" / "plots" / "presentation"


def run(cmd: list[str], cwd: Path) -> None:
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def copy_numbered_mapping(presentation_dir: Path) -> None:
    """Create friendly aliases (01_, 02_) pointing at generated files for slide building."""
    mapping = [
        ("01_hparam_grid_search_space.png", "hparam_grid_all.png"),
        ("02_hparam_grid_perplexity.png", "hparam_grid_all_perplexity.png"),
        ("03_hparam_ranked_by_loss.png", "hparam_ranked_bars.png"),
        ("04_hparam_ranked_by_perplexity.png", "hparam_ranked_bars_perplexity.png"),
    ]
    for dest_name, src_name in mapping:
        src = presentation_dir / src_name
        dest = presentation_dir / dest_name
        if src.is_file():
            shutil.copy2(src, dest)
            print(f"  Copied -> {dest.name}")


def write_readme(presentation_dir: Path, ablation_prefix: str, ablation_ok: bool) -> None:
    lines = [
        "# Presentation figures (auto-generated)",
        "",
        "## SECTION: Hyperparameters (use these PNGs)",
        "",
        "| Slide | Suggested file | What it shows |",
        "|-------|----------------|---------------|",
        "| HP-1 Grid (3 axes) | *Mermaid in `docs/presentation_hp_ab_slides.md`* — or `01_hparam_grid_search_space.png` as numeric grid | All (r × α × lr) cells; green box = best |",
        "| HP-2 Efficient search | `03_hparam_ranked_by_loss.png` or `04_hparam_ranked_by_perplexity.png` | All configs sorted; blue = best |",
        "| HP-3 Best → full FT | Same as HP-2 + say verbally: full `finetune_fireball.py` | — |",
        "",
        "**Tip:** Show **perplexity** grid (`02_...`) for intuition; **loss** for training purists — same ranking.",
        "",
        "## SECTION: Ablation study",
        "",
    ]
    if ablation_ok:
        lines.extend(
            [
                "| Slide | Suggested file | What it shows |",
                "|-------|----------------|---------------|",
                "| AB-1 Architecture | *Mermaid diagram* in docs (toggles) | — |",
                "| AB-2 Seven configs | *Table on slide* (see READMEeval) | — |",
                "| AB-3 Metrics | Icons + bullets (no single plot required) | — |",
                "| AB-4 Results | `pres_ablation_deltas_vs_baseline.png` or `pres_ablation_scores_grouped.png` | Δ vs baseline or grouped metrics |",
                "| AB-5 Takeaway | `pres_ablation_quality_vs_latency.png` | Quality vs latency trade-off |",
                "| Appendix | `pres_ablation_metric_heatmap.png` | Config × metric heatmap |",
                "",
                f"Files use prefix `pres_ablation_*` from `--ablation-prefix {ablation_prefix}`.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Ablation plots were **skipped** (no `data/" + ablation_prefix + "_*.json` files or combine failed).",
                "Run evaluations first, then:",
                f"  python scripts/evaluate_ablation_combine.py --prefix {ablation_prefix} --plots-dir data/plots/presentation --plot-prefix pres_ablation",
                "",
            ]
        )

    lines.append("Regenerate everything:")
    lines.append("  python scripts/generate_presentation_plots.py")
    lines.append("")

    (presentation_dir / "README_SLIDES.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote -> {presentation_dir / 'README_SLIDES.md'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hparam-json", type=Path, default=ROOT / "data" / "hparam_results.json")
    p.add_argument("--ablation-prefix", type=str, default="eval_results8")
    p.add_argument("--skip-ablation", action="store_true")
    p.add_argument("--skip-hparam", action="store_true")
    args = p.parse_args()

    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    if not args.skip_hparam:
        if not args.hparam_json.is_file():
            print(f"WARNING: {args.hparam_json} missing — skipping hparam plots.")
        else:
            run(
                [
                    py,
                    str(ROOT / "scripts" / "plot_hparam_results.py"),
                    "--json",
                    str(args.hparam_json),
                    "--out",
                    str(PRESENTATION_DIR),
                    "--metric",
                    "both",
                ],
                cwd=ROOT,
            )
            copy_numbered_mapping(PRESENTATION_DIR)

    ablation_ok = False
    if not args.skip_ablation:
        files = sorted((ROOT / "data").glob(f"{args.ablation_prefix}_*.json"))
        if not files:
            print(f"WARNING: No files match data/{args.ablation_prefix}_*.json — skipping ablation plots.")
        else:
            try:
                run(
                    [
                        py,
                        str(ROOT / "scripts" / "evaluate_ablation_combine.py"),
                        "--prefix",
                        args.ablation_prefix,
                        "--plot-prefix",
                        "pres_ablation",
                        "--plots-dir",
                        str(PRESENTATION_DIR),
                    ],
                    cwd=ROOT,
                )
                ablation_ok = True
            except SystemExit as e:
                print(f"WARNING: evaluate_ablation_combine failed ({e}).")

    write_readme(PRESENTATION_DIR, args.ablation_prefix, ablation_ok)
    print(f"\nDone. Figures in: {PRESENTATION_DIR}")


if __name__ == "__main__":
    main()
