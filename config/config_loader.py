"""Load settings.yaml into typed dataclasses. Validates API keys at startup."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_SETTINGS_PATH = Path(__file__).parent / "settings.yaml"


@dataclass
class ModelConfig:
    name: str
    sdk: str
    model: str
    api_key_env: str
    timeout_sec: int
    max_tokens: int
    base_url: str | None = None


@dataclass
class PromptsConfig:
    initial: str
    critique: str
    synthesis: str
    personas: dict[str, str] = field(default_factory=dict)


@dataclass
class DefaultsConfig:
    rounds: int
    max_rounds: int
    output_dir: Path
    synthesizer: str
    default_panel: list[str] = field(default_factory=list)
    full_panel: list[str] = field(default_factory=list)


@dataclass
class AppConfig:
    defaults: DefaultsConfig
    models: dict[str, ModelConfig]
    prompts: PromptsConfig
    available_providers: set[str] = field(default_factory=set)


def load_config(settings_path: Path = _SETTINGS_PATH) -> AppConfig:
    """Load and validate configuration from settings.yaml.

    Raises FileNotFoundError if settings file missing.
    Logs warnings for missing API keys but does not raise — callers check
    available_providers count.
    """
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")

    with settings_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    defaults_raw = raw["defaults"]
    defaults = DefaultsConfig(
        rounds=int(defaults_raw["rounds"]),
        max_rounds=int(defaults_raw["max_rounds"]),
        output_dir=Path(defaults_raw["output_dir"]),
        synthesizer=str(defaults_raw["synthesizer"]),
        default_panel=list(defaults_raw["default_panel"]),
        full_panel=list(defaults_raw["full_panel"]),
    )

    prompts_raw = raw["prompts"]
    personas_raw = raw.get("personas", {})
    prompts = PromptsConfig(
        initial=prompts_raw["initial"],
        critique=prompts_raw["critique"],
        synthesis=prompts_raw["synthesis"],
        personas={k: str(v) for k, v in personas_raw.items()},
    )

    models: dict[str, ModelConfig] = {}
    available_providers: set[str] = set()

    for provider_name, model_raw in raw["models"].items():
        model_cfg = ModelConfig(
            name=provider_name,
            sdk=model_raw["sdk"],
            model=model_raw["model"],
            api_key_env=model_raw["api_key_env"],
            timeout_sec=int(model_raw["timeout_sec"]),
            max_tokens=int(model_raw["max_tokens"]),
            base_url=model_raw.get("base_url"),
        )
        models[provider_name] = model_cfg

        api_key = os.environ.get(model_raw["api_key_env"], "").strip()
        if api_key:
            available_providers.add(provider_name)
            logger.info("Provider available: %s", provider_name)
        else:
            logger.info(
                "Provider skipped (no API key): %s — set %s in .env",
                provider_name,
                model_raw["api_key_env"],
            )

    return AppConfig(
        defaults=defaults,
        models=models,
        prompts=prompts,
        available_providers=available_providers,
    )
