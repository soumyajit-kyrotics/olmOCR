"""
api.py — GPT-4o API communication via OpenRouter.

Handles sending page images + anchor prompts to GPT-4o and parsing
the YAML response. Includes robust fallback parsing for cases where
the model doesn't return clean YAML.

Calibrated API parameters (matching OLMoCR's buildsilver.py):
  - temperature: 0.1 (low randomness for faithful extraction)
  - max_tokens: 6000 (enough for full-page text)
"""

import re

import requests
import yaml


def call_gpt4o(
    prompt_text: str,
    image_base64: str,
    api_key: str,
    model: str = "openai/gpt-4o",
    temperature: float = 0.1,
    max_tokens: int = 6000,
) -> dict:
    """
    Send a single page (image + anchor prompt) to GPT-4o via OpenRouter.

    Args:
        prompt_text: the full prompt string (from build_prompt())
        image_base64: base64-encoded PNG of the page (from render_pdf_to_base64png())
        api_key: OpenRouter API key
        model: model identifier (default: openai/gpt-4o)
        temperature: sampling temperature (default: 0.1 per OLMoCR)
        max_tokens: maximum response tokens (default: 6000 per OLMoCR)

    Returns:
        Parsed dict with keys: primary_language, is_rotation_valid,
        rotation_correction, is_table, is_diagram, natural_text
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()

    # Extract the content string from the API response
    content = result["choices"][0]["message"]["content"]

    # Parse the YAML response with fallback handling
    return _parse_response(content)


def _parse_response(content: str) -> dict:
    """
    Parse GPT-4o's YAML response content into a structured dict.

    Attempts multiple strategies:
      1. Direct YAML parse (ideal case)
      2. Strip code fences (```yaml ... ```) then parse
      3. Last resort: wrap the entire content as natural_text
    """
    # Strategy 1: direct YAML parse
    try:
        parsed = yaml.safe_load(content)
        if isinstance(parsed, dict) and "natural_text" in parsed:
            return parsed
    except yaml.YAMLError:
        pass

    # Strategy 2: strip code fences and try again
    fenced = re.search(r'```(?:ya?ml)?\s*\n(.*?)```', content, re.DOTALL)
    if fenced:
        try:
            parsed = yaml.safe_load(fenced.group(1))
            if isinstance(parsed, dict) and "natural_text" in parsed:
                return parsed
        except yaml.YAMLError:
            pass

    # Strategy 3: treat entire content as natural_text
    return {
        "primary_language": None,
        "is_rotation_valid": True,
        "rotation_correction": 0,
        "is_table": False,
        "is_diagram": False,
        "natural_text": content,
    }

