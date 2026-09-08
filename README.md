# Exploring AI-Driven Formative Assessment and Adaptive Feedback for Personalised Learning

> **Research Question:** How do different prompt engineering strategies affect the quality of LLM-generated formative feedback on student short answers?

## Quick Start (5 minutes)

```bash
# 1. Install Pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# 2. Clone and enter the repo
git clone https://github.com/<your-username>/mvr-prompt-engineering.git
cd mvr-prompt-engineering

# 3. Install all dependencies
pixi install

# 4. Download ASAP-SAS train.tsv from Kaggle (see Step 3 below)
#    Place it at: data/raw/train.tsv

# 5. Verify everything works (no API keys needed, $0)
pixi run dry-run

# 6. Set your API keys for real runs
cp .env.example .env
# Edit .env with your actual keys

# 7. Smoke test with real APIs (~$0.50, 24 API calls)
pixi run test

# 8. Full experiment (~$20, 5760 API calls)
pixi run pipeline
```

## What This Repo Does

This experiment compares **4 prompt strategies** for generating formative feedback on student short answers, using the **ASAP-SAS** dataset (17,000+ real student responses with human scores).

| Strategy | Code | Description |
|----------|------|-------------|
| S1 | `zero_shot` | Direct instruction, no examples |
| S2 | `zero_shot_cot` | Zero-shot + "think step by step" reasoning |
| S3 | `few_shot_rubric` | 3 graded exemplars + rubric criteria |
| S4 | `few_shot_cot_rubric` | 3 exemplars + CoT + rubric (kitchen sink) |

**Generator model:** GPT-4o-mini (held constant across all strategies)
**Evaluator model:** Claude 3.5 Sonnet (cross-model evaluation to avoid self-enhancement bias)

## Repo Structure

```
mvr-prompt-engineering/
├── pixi.toml                  # Environment & task definitions
├── .env.example               # API key template
├── configs/
│   ├── experiment.yaml        # Full experiment parameters (30 samples, 4 strategies)
│   └── smoke_test.yaml        # Minimal test parameters (4 samples, 2 strategies)
├── prompts/
│   ├── generation/            # 4 prompt templates (the independent variable)
│   │   ├── zero_shot.txt
│   │   ├── zero_shot_cot.txt
│   │   ├── few_shot_rubric.txt
│   │   └── few_shot_cot_rubric.txt
│   └── evaluation/            # 5 G-Eval dimension prompts
│       ├── accuracy.txt
│       ├── specificity.txt
│       ├── actionability.txt
│       ├── constructive_tone.txt
│       └── pedagogical_alignment.txt
├── src/
│   ├── 01_prepare_data.py     # Download & sample from ASAP-SAS
│   ├── 02_generate.py         # Run 4 strategies × N answers × 3 runs
│   ├── 03_evaluate.py         # G-Eval scoring (5 dims × all outputs)
│   ├── 04_analyse.py          # QWK, statistics, visualisations
│   └── utils.py               # Shared helpers (API clients, parsing, I/O)
├── scripts/
│   └── run_pipeline.sh        # One-click full pipeline
├── data/                      # Created by 01_prepare_data.py
├── results/                   # Created by 02-04 scripts
└── notebooks/
    └── exploration.ipynb      # Optional interactive analysis
```

## Detailed Setup Guide

### Step 1: Install Pixi

[Pixi](https://pixi.sh) is a fast, reproducible package manager that replaces conda/venv/pip. It reads `pixi.toml` and creates an identical environment on any machine.

```bash
# macOS / Linux
curl -fsSL https://pixi.sh/install.sh | bash

# Windows (PowerShell)
iwr -useb https://pixi.sh/install.ps1 | iex

# Verify
pixi --version
```

### Step 2: Get API Keys

You need **two** API keys (one for generation, one for evaluation):

| Service | Purpose | Get Key At | Free Tier? |
|---------|---------|-----------|-----------|
| **OpenAI** | GPT-4o-mini (generator) | platform.openai.com/api-keys | $5 free credit for new accounts |
| **Anthropic** | Claude 3.5 Sonnet (evaluator) | console.anthropic.com | $5 free credit for new accounts |

Copy the example env file and add your keys:

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Step 3: Get the ASAP-SAS Dataset

The dataset is hosted on Kaggle. You have two options:

**Option A — Manual download (recommended for first time):**
1. Go to https://www.kaggle.com/competitions/asap-sas/data
2. Accept the competition rules
3. Download `train.tsv` (~17MB)
4. Place it in `data/raw/train.tsv`

**Option B — Kaggle CLI (if you have kaggle.json configured):**
```bash
pixi run download-data
```

## Three Ways to Run

### 1. Dry Run — Verify pipeline, $0 cost, no API keys needed

```bash
pixi run dry-run
```

Uses mock API responses to test the entire pipeline structure: data sampling, prompt filling, response parsing, evaluation, statistical tests, and figure generation. **Start here.**

### 2. Smoke Test — Real APIs, minimal data, ~$0.50

```bash
pixi run test
```

Uses `configs/smoke_test.yaml`: 4 samples, 2 strategies, 1 run, 2 eval dimensions. Verifies your API keys work and responses parse correctly. Total: **24 API calls**.

### 3. Full Experiment — Publication-ready results, ~$20

```bash
pixi run pipeline
```

Uses `configs/experiment.yaml`: 30 samples, 4 strategies, 3 runs, 5 eval dimensions. Total: **5,760 API calls** over ~2 hours.

### Running Individual Stages

Every mode runs 4 stages in sequence. You can also run them one at a time:

```bash
# Using full config (default)
pixi run prepare      # Stage 1: Sample data
pixi run generate     # Stage 2: Generate feedback
pixi run evaluate     # Stage 3: G-Eval scoring
pixi run analyse      # Stage 4: Statistics + figures

# Using smoke test config
pixi run test-prepare
pixi run test-generate
pixi run test-evaluate
pixi run test-analyse

# Using dry-run (any stage)
pixi run dry-prepare
pixi run dry-generate
pixi run dry-evaluate
pixi run dry-analyse
```

### Step 5: Check Results

After `pixi run analyse`, find outputs in `results/`:

```
results/
├── metrics_summary.csv          # QWK, exact agreement per strategy
├── geval_scores.csv             # All G-Eval dimension scores
├── statistical_tests.txt        # Friedman + Wilcoxon test results
├── figures/
│   ├── qwk_by_strategy.png     # Bar chart of grading accuracy
│   ├── geval_radar.png          # Radar chart of 5 feedback dimensions
│   ├── geval_heatmap.png        # Heatmap: strategy × dimension
│   └── score_distribution.png   # Box plots of score distributions
└── raw/
    ├── generation_log.jsonl     # Every API call logged
    └── evaluation_log.jsonl     # Every evaluation call logged
```

## Estimated Costs & Time

| Stage | API Calls | Estimated Cost | Time |
|-------|-----------|---------------|------|
| Generate | 360 (4×30×3) | ~$1.50 | ~15 min |
| Evaluate | 5,400 (5×360×3) | ~$15–20 | ~90 min |
| **Total** | **5,760** | **~$17–22** | **~2 hours** |

## How to Modify the Experiment

All tuneable parameters live in `configs/experiment.yaml`:

```yaml
# Change sample size (more = more statistical power, more cost)
sampling:
  n_answers: 30       # Try 50 or 100 for more power
  essay_sets: [1, 2]  # Which ASAP-SAS essay sets to use

# Change number of runs for stability
runs:
  generation: 3       # Runs per strategy per answer
  evaluation: 3       # Eval runs per output per dimension

# Change models
models:
  generator: "gpt-4o-mini-2024-07-18"
  evaluator: "claude-sonnet-4-20250514"
```

## Reproducing Results

This repo is designed for reproducibility:
- **Pixi lockfile** (`pixi.lock`) pins exact package versions
- **Seed parameters** are set in all API calls (`seed=42`)
- **Temperature = 0** for deterministic outputs
- **All API responses are logged** in `results/raw/` with timestamps and model IDs
- **Random sampling uses fixed seed** (`numpy.random.seed(42)`)

To reproduce on another machine:
```bash
git clone <this-repo>
cp .env.example .env  # Add your own API keys
pixi install
pixi run pipeline
```

## Citation

If you use this experimental framework, please cite:
```bibtex
@misc{mvr_prompt_engineering_2026,
  title={Comparing Prompt Engineering Strategies for AI-Generated Formative Feedback},
  author={<Your Name>},
  year={2026},
  howpublished={\url{https://github.com/<your-username>/mvr-prompt-engineering}}
}
```
