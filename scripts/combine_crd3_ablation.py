"""
combine_crd3_ablation.py — Convenience wrapper for plots + markdown report from CRD3 ablation JSONs.

Expects per-config files from evaluate_crd3.py, e.g.:
  data/eval_crd3_ablation_<slug>.json

This calls evaluate_ablation_combine.py with CRD3-friendly defaults.

Usage:
  python scripts/combine_crd3_ablation.py
  python scripts/combine_crd3_ablation.py --prefix eval_crd3_ablation --plots-dir data/plots/crd3_ablation
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description="Combine CRD3 ablation JSONs into plots + report.")
    p.add_argument(
        "--prefix",
        type=str,
        default="eval_crd3_ablation",
        help="Must match evaluate_crd3.py --out-prefix (data/<prefix>_*.json)",
    )
    p.add_argument(
        "--baseline",
        type=str,
        default="Full Pipeline (Groq)",
        help="Experiment name for delta plots (must exist in the JSON results).",
    )
    p.add_argument(
        "--plot-prefix",
        type=str,
        default="crd3_ablation",
        help="PNG filename prefix in plots dir",
    )
    p.add_argument(
        "--plots-dir",
        type=str,
        default="",
        help="Folder for PNGs (default: data/plots/crd3_ablation)",
    )
    args = p.parse_args()

    plots_dir = args.plots_dir.strip() or str(ROOT / "data" / "plots" / "crd3_ablation")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_ablation_combine.py"),
        "--prefix",
        args.prefix,
        "--baseline",
        args.baseline,
        "--plot-prefix",
        args.plot_prefix,
        "--plots-dir",
        plots_dir,
    ]
    print("Running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
