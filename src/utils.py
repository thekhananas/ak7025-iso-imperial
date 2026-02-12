"""
Shared utilities for the MVR prompt engineering experiment.
Handles: config loading, API clients, response parsing, logging, rate limiting.
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# API libraries are imported lazily in get_*_client() to avoid
# requiring them for stages that don't make API calls (prepare, analyse).
# This lets `pixi run analyse` work without openai/anthropic installed.

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PROMPTS_DIR = ROOT / "prompts"
CONFIGS_DIR = ROOT / "configs"


def load_config(path: str | Path | None = None) -> dict:
    """
    Load experiment configuration from YAML.
    Priority: explicit path > MVR_CONFIG env var > default experiment.yaml
    """
    if path is None:
        env_path = os.environ.get("MVR_CONFIG")
        path = Path(env_path) if env_path else CONFIGS_DIR / "experiment.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    print(f"📋 Config: {Path(path).name}")
    return cfg


def is_dry_run() -> bool:
    """Check if we're in dry-run mode (no real API calls)."""
    return os.environ.get("MVR_DRY_RUN", "").lower() in ("1", "true", "yes")


def ensure_dirs():
    """Create all output directories."""
    for d in [
        DATA_DIR / "raw",
        DATA_DIR / "sampled",
        RESULTS_DIR / "raw",
        RESULTS_DIR / "figures",
    ]:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# API Clients (lazy-init singletons)
# ---------------------------------------------------------------------------
_openai_client = None
_anthropic_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        from dotenv import load_dotenv
        import openai
        load_dotenv(ROOT / ".env")
        _openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        from dotenv import load_dotenv
        import anthropic
        load_dotenv(ROOT / ".env")
        _anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _anthropic_client


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------
class RateLimiter:
    """Simple token-bucket rate limiter for API calls."""

    def __init__(self, max_rpm: int = 60):
        self.min_interval = 60.0 / max_rpm
        self.last_call = 0.0

    def wait(self):
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()


# ---------------------------------------------------------------------------
# LLM Calling
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = "gpt-4o-mini-2024-07-18",
    temperature: float = 0.0,
    max_tokens: int = 500,
    seed: int = 42,
    rate_limiter: RateLimiter | None = None,
) -> dict[str, Any]:
    """
    Call OpenAI chat completions API.
    In dry-run mode, returns a realistic mock response.
    Returns: {"text": str, "usage": dict, "model": str, "timestamp": str}
    """
    if is_dry_run():
        return _mock_openai_response(prompt, model)

    if rate_limiter:
        rate_limiter.wait()

    client = get_openai_client()
    t0 = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0,
        seed=seed,
    )

    latency = time.time() - t0
    msg = response.choices[0].message.content or ""

    return {
        "text": msg,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
        "model": response.model,
        "latency_s": round(latency, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def call_anthropic(
    prompt: str,
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.0,
    max_tokens: int = 300,
    rate_limiter: RateLimiter | None = None,
) -> dict[str, Any]:
    """
    Call Anthropic messages API.
    In dry-run mode, returns a realistic mock response.
    Returns: {"text": str, "usage": dict, "model": str, "timestamp": str}
    """
    if is_dry_run():
        return _mock_anthropic_response(prompt, model)

    if rate_limiter:
        rate_limiter.wait()

    client = get_anthropic_client()
    t0 = time.time()

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )

    latency = time.time() - t0
    text = response.content[0].text if response.content else ""

    return {
        "text": text,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "model": response.model,
        "latency_s": round(latency, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Mock responses for --dry-run mode
# ---------------------------------------------------------------------------
import random as _random

_mock_rng = _random.Random(42)


def _mock_openai_response(prompt: str, model: str) -> dict[str, Any]:
    """Generate a realistic mock generation response."""
    score = _mock_rng.randint(0, 3)
    feedback_options = [
        f"You correctly identified some key concepts. To improve, consider discussing "
        f"the specific mechanisms involved and providing more detailed examples.",
        f"Good attempt! You mentioned relevant ideas but missed some critical elements. "
        f"Review the relationship between the variables and try to be more specific.",
        f"Your answer shows partial understanding. Focus on naming the specific "
        f"scientific processes and explaining how they connect to each other.",
        f"Strong response that covers the main points. To reach the highest level, "
        f"add more precise scientific terminology and address all parts of the question.",
    ]
    reasoning = (
        "The student addresses some key concepts from the reference answer. "
        "They correctly mention one relevant factor but omit two others. "
        "The response shows partial understanding of the topic."
    )
    text = (
        f"REASONING: {reasoning}\n"
        f"SCORE: {score}\n"
        f"FEEDBACK: {_mock_rng.choice(feedback_options)}"
    )
    return {
        "text": text,
        "usage": {"prompt_tokens": 350, "completion_tokens": 120, "total_tokens": 470},
        "model": f"{model} (dry-run mock)",
        "latency_s": round(_mock_rng.uniform(0.3, 1.2), 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _mock_anthropic_response(prompt: str, model: str) -> dict[str, Any]:
    """Generate a realistic mock evaluation response."""
    # Detect which dimension is being evaluated from the prompt
    dimension = "accuracy"
    for dim in ["accuracy", "specificity", "actionability", "constructive_tone", "pedagogical_alignment"]:
        if dim.upper() in prompt:
            dimension = dim
            break

    score = _mock_rng.randint(2, 5)
    dim_key = dimension.upper() + "_SCORE"
    reasoning = (
        f"The feedback demonstrates reasonable {dimension.replace('_', ' ')}. "
        f"It addresses the student's specific answer and provides relevant guidance."
    )
    text = f"REASONING: {reasoning}\n{dim_key}: {score}"
    return {
        "text": text,
        "usage": {"input_tokens": 400, "output_tokens": 80},
        "model": f"{model} (dry-run mock)",
        "latency_s": round(_mock_rng.uniform(0.2, 0.8), 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Response Parsing
# ---------------------------------------------------------------------------
def parse_generation_response(text: str) -> dict[str, str | int | None]:
    """
    Parse structured output from generation prompts.
    Extracts SCORE, FEEDBACK, and optionally REASONING.
    """
    result: dict[str, str | int | None] = {
        "score": None,
        "feedback": None,
        "reasoning": None,
        "parse_success": False,
    }

    # Extract SCORE
    score_match = re.search(r"SCORE:\s*(\d+)", text)
    if score_match:
        result["score"] = int(score_match.group(1))

    # Extract FEEDBACK (everything after "FEEDBACK:" until end or next section)
    feedback_match = re.search(
        r"FEEDBACK:\s*(.+?)(?:\n[A-Z_]+:|$)", text, re.DOTALL
    )
    if feedback_match:
        result["feedback"] = feedback_match.group(1).strip()

    # Extract REASONING (optional, present in CoT strategies)
    reasoning_match = re.search(
        r"REASONING:\s*(.+?)(?:\nSCORE:|$)", text, re.DOTALL
    )
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    result["parse_success"] = result["score"] is not None and result["feedback"] is not None
    return result


def parse_evaluation_response(text: str, dimension: str) -> dict[str, Any]:
    """
    Parse structured output from G-Eval evaluation prompts.
    Extracts the dimension score and reasoning.
    """
    result: dict[str, Any] = {
        "dimension": dimension,
        "score": None,
        "reasoning": None,
        "parse_success": False,
    }

    # The score key varies by dimension (e.g., ACCURACY_SCORE, SPECIFICITY_SCORE)
    dim_key = dimension.upper() + "_SCORE"
    score_match = re.search(rf"{dim_key}:\s*(\d+)", text)
    if score_match:
        result["score"] = int(score_match.group(1))

    # Fallback: try generic pattern
    if result["score"] is None:
        score_match = re.search(r"_SCORE:\s*(\d+)", text)
        if score_match:
            result["score"] = int(score_match.group(1))

    # Extract reasoning
    reasoning_match = re.search(r"REASONING:\s*(.+?)(?:\n[A-Z_]+:|$)", text, re.DOTALL)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    result["parse_success"] = result["score"] is not None
    return result


# ---------------------------------------------------------------------------
# Prompt Loading
# ---------------------------------------------------------------------------
def load_prompt_template(strategy: str) -> str:
    """Load a generation prompt template by strategy name."""
    path = PROMPTS_DIR / "generation" / f"{strategy}.txt"
    return path.read_text()


def load_eval_prompt_template(dimension: str) -> str:
    """Load a G-Eval evaluation prompt template by dimension name."""
    path = PROMPTS_DIR / "evaluation" / f"{dimension}.txt"
    return path.read_text()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def append_jsonl(record: dict, filepath: Path):
    """Append a JSON record to a JSONL log file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# ASAP-SAS Metadata
# ---------------------------------------------------------------------------
# Each essay set has different score ranges and questions.
# Source: https://www.kaggle.com/competitions/asap-sas/data
ESSAY_SET_META = {
    1: {
        "score_range": (0, 3),
        "subject": "Science — Interdependence in Ecosystems",
        "question": (
            "After reading the group's procedure, describe what additional "
            "information you would need in order to replicate the experiment. "
            "Make sure to include at least three pieces of information."
        ),
        "reference_answer": (
            "Needed information includes: the type of organism studied, the ## or "
            "amount of each organism used, the size of the container/environment, "
            "the type of food, the amount of food, the daily temperature range, "
            "the light cycle conditions, the length of the experiment."
        ),
        "rubric": (
            "Score 3: Response demonstrates complete understanding — provides 3+ "
            "specific pieces of missing information with clear scientific reasoning.\n"
            "Score 2: Response demonstrates adequate understanding — provides 1-2 "
            "specific pieces of missing information.\n"
            "Score 1: Response demonstrates partial understanding — provides vague "
            "or incomplete information.\n"
            "Score 0: Response demonstrates no understanding or is off-topic."
        ),
    },
    2: {
        "score_range": (0, 3),
        "subject": "Science — Cells and Organelles",
        "question": (
            "A student squeezes a rubber ball in his hand. Describe the energy "
            "transformation that occurs and explain where the__(energy__(goes."
        ),
        "reference_answer": (
            "The student's hand uses chemical energy (from food/muscles) which is "
            "converted to mechanical energy (motion/squeezing), which is converted "
            "to heat energy (thermal energy) through friction. Some energy is also "
            "stored as elastic potential energy in the deformed ball."
        ),
        "rubric": (
            "Score 3: Response correctly identifies the full energy transformation "
            "chain: chemical → mechanical → thermal/heat, with scientific explanation.\n"
            "Score 2: Response identifies some correct energy transformations but is "
            "incomplete or partially inaccurate.\n"
            "Score 1: Response shows minimal understanding of energy transformation.\n"
            "Score 0: Response is incorrect, irrelevant, or missing."
        ),
    },
    5: {
        "score_range": (0, 3),
        "subject": "Biology — Genetics",
        "question": (
            "List and describe three processes or features that are found in "
            "eukaryotic cells but NOT in prokaryotic cells."
        ),
        "reference_answer": (
            "Features unique to eukaryotic cells include: membrane-bound nucleus "
            "containing DNA, membrane-bound organelles (mitochondria, endoplasmic "
            "reticulum, Golgi apparatus), mitosis/meiosis for cell division, linear "
            "chromosomes, larger cell size (10-100 micrometers)."
        ),
        "rubric": (
            "Score 3: Lists and describes 3 valid eukaryotic-exclusive features.\n"
            "Score 2: Lists and describes 2 valid eukaryotic-exclusive features.\n"
            "Score 1: Lists and describes 1 valid feature or lists without description.\n"
            "Score 0: No valid features listed or response is off-topic."
        ),
    },
    6: {
        "score_range": (0, 3),
        "subject": "Science — Electricity and Magnetism",
        "question": (
            "Starting with the sun, describe the energy transformations that "
            "eventually result in the emission of light from the light bulb."
        ),
        "reference_answer": (
            "The sun produces light (radiant) energy through nuclear fusion. "
            "Plants absorb light energy and convert it to chemical energy through "
            "photosynthesis. Fossil fuels (from ancient organisms) store chemical "
            "energy. Burning fossil fuels converts chemical energy to thermal energy "
            "which heats water to steam, converting to mechanical energy (turbine), "
            "then to electrical energy (generator), which is transmitted to the "
            "light bulb and converted to light and heat energy."
        ),
        "rubric": (
            "Score 3: Complete transformation chain from sun through to light bulb "
            "with correct scientific terminology.\n"
            "Score 2: Partial chain with some correct transformations but gaps.\n"
            "Score 1: Identifies only 1-2 energy types or transformations.\n"
            "Score 0: No correct transformations identified."
        ),
    },
}

# Pre-built exemplars for few-shot prompts (hand-selected from typical ASAP-SAS responses)
FEW_SHOT_EXEMPLARS = {
    1: [
        {
            "answer": "They would need to know what type of plants and animals they used, how many of each, and what size terrarium they used.",
            "score": 3,
            "feedback": "Excellent answer! You correctly identified three crucial pieces of missing information: the organism types, quantities, and container size. These are all essential for replication.",
            "reasoning": "The student lists three distinct, specific pieces of information needed for replication: organism type, quantity, and container size. Each directly addresses a gap in the experimental procedure. This meets the rubric criteria for a score of 3.",
        },
        {
            "answer": "You would need to know how much food to give them.",
            "score": 1,
            "feedback": "You identified one valid piece of information (food amount), which shows you understand the concept of replication. To improve, think about what other variables — like the types of organisms, container size, or environmental conditions — would also need to be specified.",
            "reasoning": "The student provides only one piece of information (food amount). While this is a valid replication detail, the rubric requires at least three pieces for a score of 3. One valid point earns a score of 1.",
        },
        {
            "answer": "they should do it again and see if it works",
            "score": 0,
            "feedback": "Your answer suggests repeating the experiment, but the question asks what specific information you'd need to know to replicate it. Think about the details: What organisms were used? How many? What were the environmental conditions? Try to list at least three specific details.",
            "reasoning": "The student does not identify any specific pieces of information needed for replication. Instead, they suggest simply repeating the experiment, which doesn't address the question. This earns a score of 0.",
        },
    ],
    2: [
        {
            "answer": "Chemical energy from the hand muscles is converted to mechanical energy as the ball is squeezed, and some energy is converted to heat from friction.",
            "score": 3,
            "feedback": "Great response! You correctly traced the energy transformation from chemical energy in muscles to mechanical energy in the squeezing action, and noted the conversion to thermal energy through friction. This shows strong understanding of energy conservation.",
            "reasoning": "The student correctly identifies the full chain: chemical → mechanical → thermal. They mention the source (muscles), the action (squeezing), and the dissipation mechanism (friction). This is a complete answer deserving a score of 3.",
        },
        {
            "answer": "The energy goes from his hand to the ball.",
            "score": 1,
            "feedback": "You've noted that energy transfers from hand to ball, which is a start. To improve your answer, name the specific types of energy involved — chemical energy in your muscles becomes mechanical energy when you squeeze, and some becomes heat energy. Using scientific terminology will strengthen your response.",
            "reasoning": "The student recognizes energy transfer occurs but doesn't name any energy types or describe the transformation chain. This shows minimal understanding, earning a score of 1.",
        },
        {
            "answer": "Nothing happens to the energy it just stays in the ball.",
            "score": 0,
            "feedback": "Energy is actually transformed during this process — it doesn't simply stay in one place. When you squeeze a ball, chemical energy from your muscles converts to mechanical energy (the squeezing motion) and then partly to heat energy. Review the law of conservation of energy to strengthen this concept.",
            "reasoning": "The student claims energy doesn't change, which contradicts the law of conservation of energy. No correct energy transformations are identified. Score of 0.",
        },
    ],
}
