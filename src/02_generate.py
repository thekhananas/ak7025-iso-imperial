#!/usr/bin/env python3
"""
Stage 2: Generate feedback using 4 prompt strategies.

For each (strategy × answer × run), calls GPT-4o-mini and logs the full
response. Outputs a CSV of parsed results + a JSONL raw log.

Usage:
    pixi run generate
    # or: python src/02_generate.py
"""

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils import (
    DATA_DIR,
    ESSAY_SET_META,
    FEW_SHOT_EXEMPLARS,
    RESULTS_DIR,
    RateLimiter,
    append_jsonl,
    call_openai,
    ensure_dirs,
    load_config,
    load_prompt_template,
    parse_generation_response,
)


def build_prompt(strategy: str, sample: dict) -> str:
    """
    Fill a prompt template with sample data.
    Handles both simple (zero-shot) and few-shot templates.
    """
    template = load_prompt_template(strategy)
    es = sample["essay_set"]

    # Base substitutions (present in all templates)
    subs = {
        "question": sample["question"],
        "reference_answer": sample["reference_answer"],
        "student_answer": sample["student_answer"],
        "score_min": sample["score_min"],
        "score_max": sample["score_max"],
        "rubric": sample.get("rubric", ""),
    }

    # Few-shot exemplars (only for few_shot_* strategies)
    if "few_shot" in strategy and es in FEW_SHOT_EXEMPLARS:
        exemplars = FEW_SHOT_EXEMPLARS[es]
        for i, ex in enumerate(exemplars, 1):
            subs[f"ex{i}_answer"] = ex["answer"]
            subs[f"ex{i}_score"] = ex["score"]
            subs[f"ex{i}_feedback"] = ex["feedback"]
            if "reasoning" in ex:
                subs[f"ex{i}_reasoning"] = ex["reasoning"]

    # Fill template
    prompt = template
    for key, value in subs.items():
        prompt = prompt.replace(f"{{{key}}}", str(value))

    return prompt


def main():
    config = load_config()
    ensure_dirs()

    gen_config = config["generation"]
    strategies = gen_config["strategies"]
    n_runs = gen_config["n_runs"]
    model = gen_config["model"]

    # Load sampled data
    samples_path = DATA_DIR / "sampled" / "samples.csv"
    if not samples_path.exists():
        print("ERROR: No sampled data found. Run `pixi run prepare` first.")
        sys.exit(1)

    samples_df = pd.read_csv(samples_path)
    n_samples = len(samples_df)
    total_calls = len(strategies) * n_samples * n_runs

    print(f"🚀 Generation Stage")
    print(f"   Model: {model}")
    print(f"   Strategies: {strategies}")
    print(f"   Samples: {n_samples}")
    print(f"   Runs per combo: {n_runs}")
    print(f"   Total API calls: {total_calls}")
    print()

    rate_limiter = RateLimiter(max_rpm=gen_config["rate_limit_rpm"])
    log_path = RESULTS_DIR / "raw" / "generation_log.jsonl"

    # Clear previous log
    if log_path.exists():
        log_path.unlink()

    all_results = []

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")

        for _, sample in tqdm(
            samples_df.iterrows(),
            total=n_samples,
            desc=f"  {strategy}",
        ):
            sample_dict = sample.to_dict()
            prompt = build_prompt(strategy, sample_dict)

            for run_idx in range(n_runs):
                try:
                    response = call_openai(
                        prompt=prompt,
                        model=model,
                        temperature=gen_config["temperature"],
                        max_tokens=gen_config["max_tokens"],
                        seed=gen_config["seed"] + run_idx,  # Vary seed per run
                        rate_limiter=rate_limiter,
                    )

                    parsed = parse_generation_response(response["text"])

                    record = {
                        "sample_id": sample_dict["sample_id"],
                        "essay_set": sample_dict["essay_set"],
                        "strategy": strategy,
                        "run": run_idx,
                        "human_score": sample_dict["human_score"],
                        "predicted_score": parsed["score"],
                        "feedback": parsed["feedback"],
                        "reasoning": parsed["reasoning"],
                        "parse_success": parsed["parse_success"],
                        "latency_s": response["latency_s"],
                        "model": response["model"],
                        "prompt_tokens": response["usage"]["prompt_tokens"],
                        "completion_tokens": response["usage"]["completion_tokens"],
                    }

                    all_results.append(record)

                    # Log full response for reproducibility
                    log_record = {
                        **record,
                        "raw_response": response["text"],
                        "prompt": prompt[:200] + "...",  # Truncated for log size
                        "timestamp": response["timestamp"],
                    }
                    append_jsonl(log_record, log_path)

                except Exception as e:
                    print(f"\n  ⚠️  Error on {sample_dict['sample_id']}, "
                          f"{strategy}, run {run_idx}: {e}")
                    all_results.append(
                        {
                            "sample_id": sample_dict["sample_id"],
                            "essay_set": sample_dict["essay_set"],
                            "strategy": strategy,
                            "run": run_idx,
                            "human_score": sample_dict["human_score"],
                            "predicted_score": None,
                            "feedback": None,
                            "reasoning": None,
                            "parse_success": False,
                            "latency_s": None,
                            "model": model,
                            "prompt_tokens": None,
                            "completion_tokens": None,
                            "error": str(e),
                        }
                    )

    # --- Save results ---
    results_df = pd.DataFrame(all_results)
    out_path = RESULTS_DIR / "generation_results.csv"
    results_df.to_csv(out_path, index=False)

    # --- Summary stats ---
    print(f"\n✅ Generation complete → {out_path}")
    print(f"\nParse success rate by strategy:")
    print(results_df.groupby("strategy")["parse_success"].mean().round(3))
    print(f"\nMean latency by strategy:")
    print(results_df.groupby("strategy")["latency_s"].mean().round(2))

    total_tokens = results_df["prompt_tokens"].sum() + results_df["completion_tokens"].sum()
    print(f"\nTotal tokens used: {total_tokens:,.0f}")
    print(f"Estimated cost: ${total_tokens * 0.15 / 1e6:.2f} (GPT-4o-mini pricing)")


if __name__ == "__main__":
    main()
