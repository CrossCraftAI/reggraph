"""GitHub Models provider (OpenAI-compatible).

GitHub Models gives every GitHub account free, rate-limited access to strong
models (GPT-4o, Llama, etc.) via an OpenAI-compatible API. It is the default
hosted backend for extraction, reasoning, and the public demo.

Auth uses a GitHub token. If ``GITHUB_TOKEN`` is not set, we fall back to the
token from a logged-in ``gh`` CLI, so there's usually nothing to configure.

Note: the free tier is rate-limited, so it is fine for development and a light
demo, not heavy traffic.
"""

import subprocess
import time

from .base import LLMProvider


def _gh_cli_token() -> str | None:
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


class GitHubModelsProvider(LLMProvider):
    name = "github"

    def __init__(self, model: str, token: str | None, base_url: str) -> None:
        import httpx

        token = token or _gh_cli_token()
        if not token:
            raise ValueError(
                "No GitHub token found. Set GITHUB_TOKEN, or run `gh auth login`. "
                "The token needs GitHub Models access."
            )
        self._model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._client = httpx.Client(
            timeout=120.0,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )

    def complete(self, prompt: str, *, system: str | None = None, temperature: float = 0.0) -> str:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self._model, "messages": messages, "temperature": temperature}

        # The free tier is rate-limited (~10 req/min); back off and retry on 429.
        for attempt in range(4):
            response = self._client.post(self._url, json=payload)
            if response.status_code == 429 and attempt < 3:
                retry_after = float(response.headers.get("retry-after", "8"))
                time.sleep(min(retry_after, 60.0))
                continue
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()

        response.raise_for_status()  # exhausted retries
        return ""
