#!/usr/bin/env python3
"""
Stage 1: Prepare and sample data from ASAP-SAS dataset.

Reads raw ASAP-SAS train.tsv, stratified-samples N answers per essay set,
and saves a clean CSV ready for the generation stage.

Usage:
    pixi run prepare
    # or: python src/01_prepare_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils import DATA_DIR, ESSAY_SET_META, ensure_dirs, load_config


def main():
    config = load_config()
    ensure_dirs()

    seed = config["experiment"]["seed"]
    essay_sets = config["sampling"]["essay_sets"]
    n_per_set = config["sampling"]["n_per_set"]
    stratify = config["sampling"]["stratify_by_score"]

    raw_path = DATA_DIR / "raw" / "train.tsv"

    # --- Check dataset exists ---
    if not raw_path.exists():
        print("=" * 60)
        print("ERROR: ASAP-SAS dataset not found!")
        print(f"Expected at: {raw_path}")
        print()
        print("To fix this, do ONE of the following:")
        print()
        print("  Option A — Manual download:")
        print("    1. Go to https://www.kaggle.com/competitions/asap-sas/data")
        print("    2. Accept competition rules, download train.tsv")
        print(f"    3. Place it at: {raw_path}")
        print()
        print("  Option B — Kaggle CLI:")
        print("    pixi run download-data")
        print("=" * 60)
        sys.exit(1)

    # --- Load dataset ---
    print(f"Loading ASAP-SAS from {raw_path}...")
    df = pd.read_csv(raw_path, sep="\t", encoding="latin-1")
    print(f"  Total records: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    # The ASAP-SAS dataset columns:
    # Id, EssaySet, Score1, Score2, EssayText
    # Score1 is the primary human score.
    required_cols = {"EssaySet", "Score1", "EssayText"}
    if not required_cols.issubset(set(df.columns)):
        print(f"ERROR: Expected columns {required_cols}, found {set(df.columns)}")
        sys.exit(1)

    # --- Filter to selected essay sets ---
    df_filtered = df[df["EssaySet"].isin(essay_sets)].copy()
    print(f"  After filtering to essay sets {essay_sets}: {len(df_filtered):,} records")

    # --- Stratified sampling ---
    rng = np.random.default_rng(seed)
    sampled_frames = []

    for es in essay_sets:
        subset = df_filtered[df_filtered["EssaySet"] == es]
        meta = ESSAY_SET_META.get(es, {})
        score_min, score_max = meta.get("score_range", (0, 3))

        if stratify:
            # Sample proportionally from each score level
            per_score = max(1, n_per_set // (score_max - score_min + 1))
            for score in range(score_min, score_max + 1):
                pool = subset[subset["Score1"] == score]
                n_take = min(per_score, len(pool))
                if n_take > 0:
                    chosen = pool.sample(n=n_take, random_state=int(rng.integers(1e6)))
                    sampled_frames.append(chosen)
        else:
            n_take = min(n_per_set, len(subset))
            chosen = subset.sample(n=n_take, random_state=int(rng.integers(1e6)))
            sampled_frames.append(chosen)

    sampled = pd.concat(sampled_frames, ignore_index=True)

    # --- Enrich with metadata ---
    records = []
    for _, row in sampled.iterrows():
        es = row["EssaySet"]
        meta = ESSAY_SET_META.get(es, {})
        records.append(
            {
                "sample_id": f"es{es}_{row['Id']}",
                "essay_set": es,
                "student_answer": row["EssayText"],
                "human_score": row["Score1"],
                "score_min": meta.get("score_range", (0, 3))[0],
                "score_max": meta.get("score_range", (0, 3))[1],
                "question": meta.get("question", ""),
                "reference_answer": meta.get("reference_answer", ""),
                "rubric": meta.get("rubric", ""),
                "subject": meta.get("subject", ""),
            }
        )

    result = pd.DataFrame(records)

    # --- Save ---
    out_path = DATA_DIR / "sampled" / "samples.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    print(f"\n✅ Sampled {len(result)} answers → {out_path}")
    print(f"\nBreakdown by essay set and score:")
    print(result.groupby(["essay_set", "human_score"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
