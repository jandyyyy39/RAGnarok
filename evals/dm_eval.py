import json
import argparse
from pathlib import Path
from typing import Any, Dict, List

from config import Config
from agents.dungeon_master import DMAgent
from groq import Groq
from openai import OpenAI


def build_client(use_local: bool):
    if use_local:
        print("Using local model via Ollama-compatible endpoint...")
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        model_profile = "LOCAL"
    else:
        print("Using Groq API...")
        client = Groq(api_key=Config.GROQ_API_KEY)
        model_profile = "GROQ"
    return client, model_profile


def load_cases(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "cases" not in data or not isinstance(data["cases"], list):
        raise ValueError("JSON file must contain a top-level 'cases' list.")

    return data["cases"]


def compare_tool_args(
    expected: Dict[str, Any], actual: Dict[str, Any]
) -> Dict[str, bool]:
    field_results = {}
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        field_results[key] = actual_value == expected_value
    return field_results


def evaluate_case(dm: DMAgent, case: Dict[str, Any]) -> Dict[str, Any]:
    case_id = case["id"]
    expected_mode = case["expected_mode"]

    try:
        result = dm.generate_response(
            player_input=case["player_input"],
            world_state=case["world_state"],
            ruling=case["ruling"],
        )
    except Exception as e:
        return {
            "case_id": case_id,
            "category": case.get("category"),
            "status": "error",
            "error": str(e),
            "expected_mode": expected_mode,
            "predicted_mode": None,
            "mode_correct": False,
            "tool_args_exact_match": False,
            "tool_arg_field_results": {},
            "raw_result": None,
        }

    predicted_mode = result.get("type")
    mode_correct = predicted_mode == expected_mode

    tool_args_exact_match = None
    tool_arg_field_results = {}

    if expected_mode == "tool_call":
        expected_tool_args = case.get("expected_tool_args", {})
        actual_tool_args = result.get("action", {}) if predicted_mode == "tool_call" else {}
        tool_arg_field_results = compare_tool_args(expected_tool_args, actual_tool_args)
        tool_args_exact_match = (
            predicted_mode == "tool_call" and all(tool_arg_field_results.values())
        )

    return {
        "case_id": case_id,
        "category": case.get("category"),
        "status": "ok",
        "expected_mode": expected_mode,
        "predicted_mode": predicted_mode,
        "mode_correct": mode_correct,
        "tool_args_exact_match": tool_args_exact_match,
        "tool_arg_field_results": tool_arg_field_results,
        "raw_result": result,
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    ok_results = [r for r in results if r["status"] == "ok"]
    error_results = [r for r in results if r["status"] == "error"]

    mode_correct_count = sum(1 for r in ok_results if r["mode_correct"])
    mode_accuracy = mode_correct_count / total if total else 0.0

    tool_cases = [
        r for r in ok_results
        if r["expected_mode"] == "tool_call"
    ]
    tool_exact_matches = sum(
        1 for r in tool_cases if r["tool_args_exact_match"] is True
    )
    tool_arg_accuracy = (
        tool_exact_matches / len(tool_cases) if tool_cases else None
    )

    field_totals: Dict[str, int] = {}
    field_correct: Dict[str, int] = {}

    for r in tool_cases:
        for field, is_correct in r["tool_arg_field_results"].items():
            field_totals[field] = field_totals.get(field, 0) + 1
            field_correct[field] = field_correct.get(field, 0) + int(is_correct)

    field_accuracy = {}
    for field in field_totals:
        field_accuracy[field] = field_correct[field] / field_totals[field]

    by_category: Dict[str, Dict[str, Any]] = {}
    for r in ok_results:
        cat = r.get("category", "unknown")
        by_category.setdefault(cat, {"total": 0, "mode_correct": 0})
        by_category[cat]["total"] += 1
        by_category[cat]["mode_correct"] += int(r["mode_correct"])

    for cat, stats in by_category.items():
        stats["mode_accuracy"] = (
            stats["mode_correct"] / stats["total"] if stats["total"] else 0.0
        )

    return {
        "total_cases": total,
        "successful_runs": len(ok_results),
        "errors": len(error_results),
        "mode_accuracy": mode_accuracy,
        "tool_case_count": len(tool_cases),
        "tool_arg_exact_match_accuracy": tool_arg_accuracy,
        "tool_arg_field_accuracy": field_accuracy,
        "by_category": by_category,
    }


def main():
    parser = argparse.ArgumentParser(description="Run DM evaluation cases.")
    parser.add_argument(
        "--cases",
        type=str,
        default="evals/dm_eval_cases.json",
        help="Path to the JSON file containing evaluation cases.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evals/dm_eval_results.json",
        help="Path to save detailed results JSON.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local model instead of Groq.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for number of cases to run.",
    )
    args = parser.parse_args()

    cases_path = Path(args.cases)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases = load_cases(cases_path)
    if args.limit is not None:
        cases = cases[:args.limit]

    client, model_profile = build_client(args.local)
    dm = DMAgent(client=client, model_profile=model_profile)

    results = []
    print(f"Running {len(cases)} test case(s)...")

    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] {case['id']} ({case.get('category', 'unknown')})")
        result = evaluate_case(dm, case)
        results.append(result)

        if result["status"] == "error":
            print(f"  ERROR: {result['error']}")
        else:
            print(
                f"  expected={result['expected_mode']} "
                f"predicted={result['predicted_mode']} "
                f"mode_correct={result['mode_correct']}"
            )
            if result["expected_mode"] == "tool_call":
                print(f"  tool_args_exact_match={result['tool_args_exact_match']}")

    summary = summarize(results)

    payload = {
        "summary": summary,
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY ===")
    print(f"Total cases: {summary['total_cases']}")
    print(f"Successful runs: {summary['successful_runs']}")
    print(f"Errors: {summary['errors']}")
    print(f"Mode accuracy: {summary['mode_accuracy']:.2%}")

    if summary["tool_arg_exact_match_accuracy"] is not None:
        print(
            "Tool arg exact-match accuracy: "
            f"{summary['tool_arg_exact_match_accuracy']:.2%}"
        )

    if summary["tool_arg_field_accuracy"]:
        print("Tool arg field accuracy:")
        for field, acc in summary["tool_arg_field_accuracy"].items():
            print(f"  {field}: {acc:.2%}")

    print("\nBy category:")
    for cat, stats in summary["by_category"].items():
        print(
            f"  {cat}: total={stats['total']} "
            f"mode_accuracy={stats['mode_accuracy']:.2%}"
        )

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()