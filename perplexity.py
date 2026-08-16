"""
Token-probability signal using GPT-2. This is the piece the brief points
you toward directly: "run text through a small local model for token
probabilities, then do your own analysis on those numbers." The model
only produces numbers here — it never renders a verdict. The verdict
logic stays in app.py, same as your other signals.

Install once:
    pip install torch transformers

First run downloads gpt2 (~500MB) from Hugging Face and caches it locally
under ~/.cache/huggingface — subsequent runs are fast and offline.
"""

import math
import torch
from functools import lru_cache
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

_MODEL_NAME = "gpt2"


@lru_cache(maxsize=1)
def _load_model():
    """Loaded once per process — Streamlit re-runs the script on every
    interaction, so caching this is what keeps the app usable."""
    tokenizer = GPT2TokenizerFast.from_pretrained(_MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(_MODEL_NAME)
    model.eval()
    return tokenizer, model


def sentence_perplexity(sentence: str) -> float | None:
    """
    Perplexity = how 'surprised' GPT-2 is by this sentence, on average,
    per token. Lower perplexity = more predictable = more machine-like.
    Human writing tends to run higher and more variable; machine writing
    tends to cluster lower and tighter.

    Returns None for sentences too short to score meaningfully (GPT-2
    needs at least 2 tokens to compute a next-token loss).
    """
    tokenizer, model = _load_model()
    encodings = tokenizer(sentence, return_tensors="pt")
    input_ids = encodings["input_ids"]

    if input_ids.shape[1] < 2:
        return None

    with torch.no_grad():
        # labels=input_ids makes HF compute next-token cross-entropy loss
        # for us — this IS the "token probability" analysis the brief
        # describes, not a generated verdict.
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return math.exp(loss.item())


def essay_perplexity_profile(sentences: list[str]) -> dict:
    """
    Per-sentence perplexity plus the essay-level mean and variance.
    Variance matters as much as the mean: a human essay's perplexity
    swings sentence to sentence; heavily-polished or machine text tends
    to sit in a narrow band. This mirrors your existing burstiness (CV)
    signal, but built from model probabilities instead of sentence length.
    """
    scores = []
    for s in sentences:
        ppl = sentence_perplexity(s)
        if ppl is not None:
            scores.append(ppl)

    if not scores:
        return {"per_sentence": [], "mean": None, "cv": None}

    mean = sum(scores) / len(scores)
    variance = sum((x - mean) ** 2 for x in scores) / len(scores)
    std_dev = variance ** 0.5
    cv = std_dev / mean if mean > 0 else 0.0

    return {
        "per_sentence": scores,
        "mean": round(mean, 2),
        "cv": round(cv, 3),
    }