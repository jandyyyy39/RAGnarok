"""
Merge per-experiment eval JSONs from two prefixes into one, so evaluate_ablation_combine.py
can plot Groq ablations + new Local FT (e.g. CRD3) on one chart.

Example:
  python scripts/merge_ablation_eval_for_plots.py \\
    --out-prefix eval_merged_crd3 \\
    --from-prefix eval_results8 --slugs full_pipeline_groq,no_rag,no_memory,no_npc_pass,bare_dm_only \\
    --from-prefix eval_crd3_ft --slugs local_ft_rag_on,local_ft_no_rag

Then:
  python scripts/evaluate_ablation_combine.py --prefix eval_merged_crd3 \\
    --baseline "Full Pipeline (Groq)" --plot-prefix crd3_merged_ablation --plots-dir data/plots/crd3_merged

All merged files must use the SAME number of eval inputs (e.g. all --quick = 8, or all -n 25).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def main() -> None:
    p = argparse.ArgumentParser(description="Merge eval JSONs under one prefix for ablation plots.")
    p.add_argument("--out-prefix", required=True, help="Output prefix, e.g. eval_merged_crd3")
    p.add_argument(
        "--from-prefix",
        required=True,
        action="append",
        metavar="PREFIX",
        help="Source prefix (repeat for multiple sources): data/<PREFIX>_<slug>.json",
    )
    p.add_argument(
        "--slugs",
        required=True,
        action="append",
        metavar="LIST",
        help="Comma-separated slugs for the PREVIOUS --from-prefix in order (repeat per prefix).",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if len(args.from_prefix) != len(args.slugs):
        sys.exit("ERROR: repeat --from-prefix and --slugs together (one --slugs per --from-prefix).")

    out_prefix = args.out_prefix.strip()
    planned: list[tuple[Path, Path]] = []

    for fp, slug_list in zip(args.from_prefix, args.slugs):
        prefix = fp.strip()
        for slug in slug_list.split(","):
            slug = slug.strip()
            if not slug:
                continue
            src = DATA / f"{prefix}_{slug}.json"
            dst = DATA / f"{out_prefix}_{slug}.json"
            if not src.is_file():
                print(f"ERROR: missing {src}", file=sys.stderr)
                sys.exit(1)
            planned.append((src, dst))

    counts: list[tuple[str, int]] = []
    for src, _ in planned:
        with open(src, encoding="utf-8") as f:
            data = json.load(f)
        n = len(data) if isinstance(data, list) else 0
        counts.append((src.name, n))

    uniq = {n for _, n in counts}
    if len(uniq) > 1:
        print("WARNING: different row counts — comparison may be unfair:", file=sys.stderr)
        for name, n in counts:
            print(f"  {n:3d}  {name}", file=sys.stderr)

    for src, dst in planned:
        print(f"  {src.name} -> {dst.name}")
        if not args.dry_run:
            shutil.copy2(src, dst)

    if not args.dry_run:
        print(
            f"\nNext:\n  python scripts/evaluate_ablation_combine.py --prefix {out_prefix} \\\n"
            f'    --baseline "Full Pipeline (Groq)" --plot-prefix crd3_merged_ablation --plots-dir data/plots/crd3_merged'
        )


if __name__ == "__main__":
    main()
