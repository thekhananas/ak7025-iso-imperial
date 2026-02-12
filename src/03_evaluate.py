#!/usr/bin/env python3
"""
Stage 3: Evaluate generated feedback using G-Eval (LLM-as-a-Judge).

For each generated feedback, evaluates 5 quality dimensions using Claude 3.5
Sonnet as the evaluator (cross-model evaluation to avoid self-enhancement bias).

Usage:
    pixi run evaluate
    # or: python src/03_evaluate.py
"""

import sys

import pandas as pd
from tqdm import tqdm

from utils import (
    DATA_DIR,
    RESULTS_DIR,
    RateLimiter,
    append_jsonl,
    call_anthropic,
    ensure_dirs,
    load_config,
    load_eval_prompt_template,
    parse_evaluation_response,
)


def build_eval_prompt(dimension: str, sample: dict, feedback: str) -> str:
    """Fill an evaluation prompt template with sample data and feedback."""
    template = load_eval_prompt_template(dimension)

    subs = {
        "question": sample["question"],
        "reference_answer": sample["reference_answer"],
        "student_answer": sample["student_answer"],
        "feedback": feedback,
    }

    prompt = template
    for key, value in subs.items():
        prompt = prompt.replace(f"{{{key}}}", str(value))

    return prompt


def main():
    config = load_config()
    ensure_dirs()

    eval_config = config["evaluation"]
    dimensions = eval_config["dimensions"]
    n_eval_runs = eval_config["n_runs"]
    model = eval_config["model"]

    # Load generation results
    gen_path = RESULTS_DIR / "generation_results.csv"
    if not gen_path.exists():
        print("ERROR: No generation results found. Run `pixi run generate` first.")
        sys.exit(1)

    gen_df = pd.read_csv(gen_path)

    # Load original samples for question/reference context
    samples_df = pd.read_csv(DATA_DIR / "sampled" / "samples.csv")
    samples_lookup = samples_df.set_index("sample_id").to_dict("index")

    # Filter to successfully parsed outputs only
    valid_gen = gen_df[gen_df["parse_success"] == True].copy()
    n_valid = len(valid_gen)
    total_calls = n_valid * len(dimensions) * n_eval_runs

    print(f"🔍 Evaluation Stage (G-Eval)")
    print(f"   Evaluator model: {model}")
    print(f"   Dimensions: {dimensions}")
    print(f"   Valid generation outputs: {n_valid}")
    print(f"   Eval runs per (output, dimension): {n_eval_runs}")
    print(f"   Total API calls: {total_calls}")
    print(f"   Estimated cost: ~${total_calls * 0.004:.2f}")
    print()

    rate_limiter = RateLimiter(max_rpm=eval_config["rate_limit_rpm"])
    log_path = RESULTS_DIR / "raw" / "evaluation_log.jsonl"

    if log_path.exists():
        log_path.unlink()

    all_evals = []

    for dim in dimensions:
        print(f"\n--- Dimension: {dim} ---")

        for _, gen_row in tqdm(
            valid_gen.iterrows(),
            total=n_valid,
            desc=f"  {dim}",
        ):
            sample_id = gen_row["sample_id"]
            sample = samples_lookup.get(sample_id, {})

            if not sample:
                continue

            feedback_text = gen_row["feedback"]
            if pd.isna(feedback_text) or not feedback_text:
                continue

            eval_prompt = build_eval_prompt(dim, sample, feedback_text)

            for eval_run in range(n_eval_runs):
                try:
                    response = call_anthropic(
                        prompt=eval_prompt,
                        model=model,
                        temperature=eval_config["temperature"],
                        max_tokens=eval_config["max_tokens"],
                        rate_limiter=rate_limiter,
                    )

                    parsed = parse_evaluation_response(response["text"], dim)

                    record = {
                        "sample_id": sample_id,
                        "strategy": gen_row["strategy"],
                        "gen_run": gen_row["run"],
                        "dimension": dim,
                        "eval_run": eval_run,
                        "eval_score": parsed["score"],
                        "eval_reasoning": parsed["reasoning"],
                        "parse_success": parsed["parse_success"],
                        "eval_model": response["model"],
                        "eval_latency_s": response["latency_s"],
                    }

                    all_evals.append(record)

                    log_record = {
                        **record,
                        "raw_response": response["text"],
                        "timestamp": response["timestamp"],
                    }
                    append_jsonl(log_record, log_path)

                except Exception as e:
                    print(f"\n  ⚠️  Error evaluating {sample_id}, {dim}, "
                          f"run {eval_run}: {e}")
                    all_evals.append(
                        {
                            "sample_id": sample_id,
                            "strategy": gen_row["strategy"],
                            "gen_run": gen_row["run"],
                            "dimension": dim,
                            "eval_run": eval_run,
                            "eval_score": None,
                            "eval_reasoning": None,
                            "parse_success": False,
                            "eval_model": model,
                            "eval_latency_s": None,
                            "error": str(e),
                        }
                    )

    # --- Save results ---
    eval_df = pd.DataFrame(all_evals)
    out_path = RESULTS_DIR / "evaluation_results.csv"
    eval_df.to_csv(out_path, index=False)

    # --- Summary ---
    print(f"\n✅ Evaluation complete → {out_path}")
    print(f"\nParse success rate: {eval_df['parse_success'].mean():.1%}")
    print(f"\nMean scores by strategy × dimension:")

    valid_evals = eval_df[eval_df["parse_success"] == True]
    if len(valid_evals) > 0:
        pivot = valid_evals.groupby(["strategy", "dimension"])["eval_score"].mean()
        print(pivot.unstack().round(2))


if __name__ == "__main__":
    main()
