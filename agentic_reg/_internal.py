"""Small helpers used across the package — nothing domain-specific lives here."""

import json
import re


def parse_json_object(raw: str) -> dict:
    """Pull the first JSON object out of a string that may contain extra prose.

    LLMs often wrap JSON in markdown fences or prefix it with explanatory text.
    This is a permissive extractor that survives those wrappers.
    """
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
