"""Token / cost accounting.

Prices (USD per 1M tokens) for gpt-6-astra from the OpenAI model page as recorded in
research_notes/.../seed_papers.md Q8 (2026-10): input 10, cached input 1, cache write 12.5, output 50;
prompts > 272K input tokens are billed 2x input/cached and 1.5x output. usage.input_tokens_details has
cached_tokens and (openai SDK >= 3.x) cache_write_tokens; both are priced separately when present.
Reasoning tokens are billed as output tokens (they are included in output_tokens).
"""
import math

PRICES = {
    "gpt-6-astra": dict(input=10.0, cached=1.0, cache_write=12.5, output=50.0, long_threshold=272_000, long_in=2.0, long_out=1.5),
}


def call_cost(usage, model="gpt-6-astra"):
    p = PRICES.get(model)
    if p is None or not usage:
        return None
    inp = usage.get("input_tokens", 0) or 0
    cached = usage.get("cached_tokens", 0) or 0
    write = usage.get("cache_write_tokens", 0) or 0  # reported by newer SDKs / API versions
    out = usage.get("output_tokens", 0) or 0
    li, lo = (p["long_in"], p["long_out"]) if inp > p["long_threshold"] else (1.0, 1.0)
    plain = max(inp - cached - write, 0)
    return ((plain * p["input"] + write * p["cache_write"] + cached * p["cached"]) * li + out * p["output"] * lo) / 1e6


def image_tokens(w, h, patch=32, per_patch=1.0, overhead=0):
    """ASSUMPTION (not verified for gpt-6-astra): patch-based image accounting as for the GPT-5 family,
    one token per 32x32 patch (~256 tokens for 512x512). Real numbers come from response.usage."""
    return int(math.ceil(w / patch) * math.ceil(h / patch) * per_patch + overhead)


def estimate_request_tokens(stats, image_res=512):
    """Rough input-token estimate of one request from OpenAIResponsesBackend.request_stats()."""
    return int(stats.get("text_chars", 0) / 4 + stats.get("n_images", 0) * image_tokens(image_res, image_res))
