"""
evaluate_crd3.py — Ablation evaluation on the CRD3 eval split (NOT FIREBALL).

Reads ground-truth references from Config.CRD3_EVAL_FILE (default: data/crd3_eval.jsonl).
Uses the same 7 ablation configurations as scripts/evaluate.py.

Outputs per-config JSON files:
  data/<out-prefix>_<slug>.json

Run ALL ablations (sequential):
  python scripts/evaluate_crd3.py --quick
  python scripts/evaluate_crd3.py -n 25

Run ONE config:
  python scripts/evaluate_crd3.py --experiment "Local FT (RAG On)" --fresh

Then combine + plots:
  python scripts/combine_crd3_ablation.py

LLM-as-judge (qualitative) is ON by default — requires GROQ_API_KEY.
Quantitative-only (faster/cheaper): add --no-llm-judge
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from config import Config

import evaluate as ev


def load_crd3_test_inputs(eval_path: Path, limit: int) -> list[dict]:
    """Load input / reference / type from CRD3 JSONL (instruction, input, output)."""
    if not eval_path.is_file():
        print(f"ERROR: CRD3 eval file not found: {eval_path}")
        sys.exit(1)

    samples: list[dict] = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            inp = row.get("input", "")
            ref = row.get("output", "")
            itype = row.get("type") or ev._infer_input_type(inp)
            samples.append({"input": inp, "reference": ref, "type": itype})
            if len(samples) >= limit:
                break

    print(f"Loaded {len(samples)} test inputs from CRD3 eval: {eval_path}")
    return samples


def match_experiments(query: str) -> list[dict]:
    q = query.strip().lower()
    matches = []
    for e in ev.EXPERIMENTS:
        if (
            e["name"].lower() == q
            or ev.experiment_slug(e["name"]).startswith(q)
            or q in e["name"].lower()
        ):
            matches.append(e)
    return matches


def run_one_or_all(
    experiments: list[dict],
    inputs: list[dict],
    out_prefix: str,
    use_llm: bool,
    fresh: bool,
    extra_mode: str,
) -> None:
    ev.DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_loaded: list[dict] = []

    for e in experiments:
        per_out = ev.DATA_DIR / f"{out_prefix}_{ev.experiment_slug(e['name'])}.json"
        print(f"\n[Output] {per_out}")
        if fresh and per_out.exists():
            per_out.unlink()

        loaded = ev.run_experiment_to_file(
            test_inputs=inputs,
            exp=e,
            use_llm_judge=use_llm,
            out_file=per_out,
            resume=not fresh,
            extra_metrics_mode=extra_mode,
        )
        all_loaded.extend(loaded)

    if all_loaded:
        ev.print_summary(all_loaded)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGnarok ablation evaluation on CRD3 eval set (crd3_eval.jsonl)."
    )
    parser.add_argument(
        "--eval-jsonl",
        type=Path,
        default=None,
        help=f"Path to CRD3 eval JSONL (default: {Config.CRD3_EVAL_FILE})",
    )
    parser.add_argument("--quick", action="store_true", help="Use first 8 eval lines only")
    parser.add_argument("--n", type=int, default=25, help="Number of eval lines (ignored if --quick)")
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Skip LLM-as-judge (default for CRD3 is ON: narrative_quality, rules_accuracy, character_voice, relevance + llm_composite)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help='Run a single ablation (exact name, substring, or slug). Omit to run ALL 7 configs.',
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="eval_crd3_ablation",
        help="Output files: data/<prefix>_<slug>.json",
    )
    parser.add_argument("--fresh", action="store_true", help="Overwrite output (no resume)")
    parser.add_argument(
        "--extra-metrics-mode",
        type=str,
        default="all",
        choices=["all", "bleu_chrf_only", "none"],
        help="Overlap / diversity metrics (same as evaluate.py)",
    )
    parser.add_argument("--no-extra-metrics", action="store_true", help="Same as --extra-metrics-mode none")
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="Print ablation names and exit",
    )
    args = parser.parse_args()
    # CRD3 eval: qualitative + quantitative by default (matches project rubric).
    use_llm = not args.no_llm_judge

    if args.list_experiments:
        for e in ev.EXPERIMENTS:
            print(f"  {e['name']:<28}  slug: {ev.experiment_slug(e['name'])}")
        sys.exit(0)

    limit = 8 if args.quick else args.n
    eval_path = Path(args.eval_jsonl) if args.eval_jsonl else Config.CRD3_EVAL_FILE
    inputs = load_crd3_test_inputs(eval_path, limit)

    # Avoid merging partial results from legacy FIREBALL eval_results.json
    ev.RESULTS_FILE = ev.DATA_DIR / "eval_crd3_legacy_merge_disabled.json"

    extra_mode = "none" if args.no_extra_metrics else args.extra_metrics_mode

    if args.experiment:
        matches = match_experiments(args.experiment)
        if not matches:
            print(f"ERROR: No experiment matches --experiment={args.experiment!r}")
            print("Use --list-experiments")
            sys.exit(1)
        if len(matches) > 1:
            print(f"WARNING: multiple matches for '{args.experiment}'. Running {len(matches)} configs.")
    else:
        matches = list(ev.EXPERIMENTS)

    print(f"\nCRD3 eval: {len(matches)} config(s) × {len(inputs)} inputs")
    print(f"  Eval file: {eval_path}")
    print(f"  Out prefix: {args.out_prefix}")
    print("  Quantitative: ROUGE / BERTScore / BLEU / chrF + rule coverage + latency")
    if use_llm:
        print("  LLM-as-judge: ON (narrative, rules, voice, relevance + composite) — needs GROQ_API_KEY\n")
    else:
        print("  LLM-as-judge: OFF (--no-llm-judge)\n")

    run_one_or_all(matches, inputs, args.out_prefix, use_llm, args.fresh, extra_mode)


if __name__ == "__main__":
    main()
