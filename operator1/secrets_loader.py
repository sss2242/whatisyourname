"""Load API secrets from Kaggle, .env file, or environment variables.

On Kaggle, keys are stored via the Kaggle Secrets client.  For local
development, falls back to a ``.env`` file in the project root, then
to environment variables.

ALL keys and identities listed below are REQUIRED.  The pipeline will
refuse to start if any are missing.  Users must register (free) at
each API provider and populate their ``.env`` file before running.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Attempt to load .env file for local development
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Load variables from .env file if python-dotenv is available."""
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]
        load_dotenv(env_path)
        logger.info("Loaded environment from %s", env_path)
    except ImportError:
        # Manual fallback: parse simple KEY=VALUE lines
        try:
            with open(env_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ.setdefault(key, value)
            logger.info("Loaded .env file manually (python-dotenv not installed)")
        except Exception as exc:
            logger.warning("Failed to parse .env file: %s", exc)


# All keys -- pipeline will halt if any are missing.
# Each key maps to a human-readable description for error messages.
_REQUIRED_KEYS: dict[str, str] = {
    # LLM provider (at least one required for report generation)
    "GEMINI_API_KEY":          "LLM report generation (Google Gemini) -- https://aistudio.google.com/app/apikey",
    "ANTHROPIC_API_KEY":       "LLM report generation (Anthropic Claude) -- https://console.anthropic.com/",
    "OPENROUTER_API_KEY":      "LLM report generation (OpenRouter, 200+ models) -- https://openrouter.ai/",
    # US market
    "EDGAR_IDENTITY":          "US SEC EDGAR email identity (required by SEC regulation) -- any valid email",
    # UK market
    "COMPANIES_HOUSE_API_KEY": "UK Companies House -- https://developer.company-information.service.gov.uk/",
    # Japan market
    "JQUANTS_API_KEY":         "Japan J-Quants (free email registration) -- https://jpx-jquants.com/login",
    # South Korea market
    "DART_API_KEY":            "South Korea DART -- https://opendart.fss.or.kr/",
    # Macro data
    "FRED_API_KEY":            "US FRED macro data -- https://fred.stlouisfed.org/docs/api/api_key.html",
    "BANXICO_TOKEN":           "Mexico Banxico macro data -- https://www.banxico.org.mx/SieAPIRest/",
    # Identifier resolution
    "openfigi_key":            "OpenFIGI identifier resolution -- https://www.openfigi.com/api",
}


def _load_from_kaggle() -> dict[str, str]:
    """Attempt to read secrets via Kaggle UserSecretsClient."""
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-untyped]
        client = UserSecretsClient()
        secrets: dict[str, str] = {}
        for key in _REQUIRED_KEYS:
            try:
                value = client.get_secret(key)
                if value:
                    secrets[key] = value.strip()
            except Exception:
                pass
        return secrets
    except ImportError:
        return {}


def _load_from_env() -> dict[str, str]:
    """Fallback: read from OS environment variables."""
    secrets: dict[str, str] = {}
    for key in _REQUIRED_KEYS:
        value = os.environ.get(key)
        if value:
            secrets[key] = value.strip()
    return secrets


def load_secrets() -> dict[str, str]:
    """Return a dict of API secrets.

    Priority order:
    1. Kaggle Secrets client (if running on Kaggle)
    2. ``.env`` file in project root
    3. OS environment variables

    Does NOT validate completeness -- call ``validate_secrets()``
    separately in your entry point (main.py, run.py) to enforce that
    all required keys are present before the pipeline runs.

    **Multi-key support**: For LLM providers (GEMINI_API_KEY,
    ANTHROPIC_API_KEY), comma-separated values are preserved as-is.
    Use ``get_key_pool()`` to split them into lists for rotation.
    """
    # Try .env file first (for local development)
    _load_dotenv()

    # Try Kaggle secrets first
    secrets = _load_from_kaggle()
    if not secrets:
        secrets = _load_from_env()

    # Also collect numbered key variants (e.g. GEMINI_API_KEY_1, _2, _3)
    for provider_key in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        extra_keys = []
        for i in range(1, 10):
            numbered = os.environ.get(f"{provider_key}_{i}", "").strip()
            if numbered:
                extra_keys.append(numbered)
        if extra_keys:
            base = secrets.get(provider_key, "")
            all_keys = ([base] if base else []) + extra_keys
            secrets[provider_key] = ",".join(all_keys)

    return secrets


def get_key_pool(secrets: dict[str, str], key_name: str) -> list[str]:
    """Split a secret value into a list of keys (for multi-key rotation).

    Supports comma-separated values in a single env var or numbered
    variants (KEY_1, KEY_2, etc.) which are merged by ``load_secrets()``.

    Parameters
    ----------
    secrets:
        Dict from ``load_secrets()``.
    key_name:
        Secret key name (e.g. "GEMINI_API_KEY", "ANTHROPIC_API_KEY").

    Returns
    -------
    List of individual API keys (empty list if none found).
    """
    raw = secrets.get(key_name, "")
    if not raw:
        return []
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    # Filter out obvious placeholders
    return [k for k in keys if k and not k.startswith("placeholder")]


def validate_secrets(secrets: dict[str, str]) -> None:
    """Validate that ALL required API keys and identities are present.

    Call this from your entry point (main.py, run.py) after loading
    secrets.  Raises ``SystemExit`` with a clear message listing every
    missing key and where to register for it (all free).
    """
    missing = [
        f"  - {key}: {desc}"
        for key, desc in _REQUIRED_KEYS.items()
        if not secrets.get(key)
    ]
    if missing:
        raise SystemExit(
            "Missing required API keys / identities.\n"
            "Create a .env file from .env.example and fill in ALL values.\n"
            "All registrations are FREE.\n\n"
            "Missing keys:\n" + "\n".join(missing) + "\n"
        )

    logger.info("All %d required API keys loaded successfully.", len(_REQUIRED_KEYS))
