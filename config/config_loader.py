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
    cost_per_1m_input: float = 0.0   # USD per 1M input tokens
    cost_per_1m_output: float = 0.0  # USD per 1M output tokens


@dataclass
class PromptsConfig:
    initial: str
    critique: str
    synthesis: str
    personas: dict[str, str] = field(default_factory=dict)


@dataclass
class InboxConfig:
    dir: Path
    archive_dir: Path
    downloads_dir: Path = field(default_factory=lambda: Path.home() / "Downloads")
    scan_downloads: bool = True
    council_frontmatter_keys: list[str] = field(
        default_factory=lambda: ["mode", "rounds", "models", "synthesizer", "full"]
    )


@dataclass
class DefaultsConfig:
    rounds: int
    max_rounds: int
    output_dir: Path
    synthesizer: str
    default_panel: list[str] = field(default_factory=list)
    full_panel: list[str] = field(default_factory=list)


@dataclass
class ModeConfig:
    description: str
    emoji: str
    aliases: list[str]
    default: bool
    max_rounds: int
    token_budget: int
    round1_header: str = ""
    round1_instruction: str = ""
    round1_structure: str = ""
    round2_instruction: str = ""
    synthesis_output: str = ""

    @property
    def uses_existing_prompts(self) -> bool:
        """True for pick mode — delegates to prompts.initial/critique/synthesis."""
        return not self.round1_instruction.strip()


@dataclass
class ResearchProviderConfig:
    name: str
    model: str
    api_key_env: str
    timeout_sec: int
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0
    base_url: str | None = None


@dataclass
class ResearchConfig:
    default_providers: list[str]
    deep_providers: list[str]
    cache_dir: Path
    cache_ttl_days: int
    summary_max_tokens: int
    summary_model: str
    providers: dict[str, ResearchProviderConfig] = field(default_factory=dict)


@dataclass
class AppConfig:
    defaults: DefaultsConfig
    models: dict[str, ModelConfig]
    prompts: PromptsConfig
    inbox: InboxConfig = field(default_factory=lambda: InboxConfig(Path("./council_inbox"), Path("./council_inbox/archive")))
    available_providers: set[str] = field(default_factory=set)
    modes: dict[str, ModeConfig] = field(default_factory=dict)
    persona_mode_directives: dict[str, dict[str, str]] = field(default_factory=dict)
    research: ResearchConfig | None = None


def resolve_mode(mode_arg: str, modes: dict[str, ModeConfig]) -> str:
    """Resolve a mode name or alias to canonical mode key.

    Returns the canonical key (e.g. "pick") or raises ValueError.
    """
    if mode_arg in modes:
        return mode_arg
    for key, cfg in modes.items():
        if mode_arg in cfg.aliases:
            return key
    raise ValueError(
        f"Unknown mode '{mode_arg}'. Valid modes: {sorted(modes)} "
        f"and their aliases."
    )


def default_mode(modes: dict[str, ModeConfig]) -> str:
    """Return the canonical key of the mode marked default: true."""
    for key, cfg in modes.items():
        if cfg.default:
            return key
    raise ValueError("No default mode configured in settings.yaml")


def _validate_modes(modes: dict[str, ModeConfig]) -> None:
    """Validate mode config invariants. Raises ValueError on violation."""
    defaults = [k for k, v in modes.items() if v.default]
    if len(defaults) != 1:
        raise ValueError(
            f"Exactly one mode must have default: true, found: {defaults}"
        )

    seen_aliases: dict[str, str] = {}
    for key, cfg in modes.items():
        for alias in cfg.aliases:
            if alias != alias.lower():
                raise ValueError(
                    f"Mode '{key}' alias '{alias}' must be lowercase"
                )
            if alias in seen_aliases:
                raise ValueError(
                    f"Duplicate alias '{alias}' in modes '{seen_aliases[alias]}' and '{key}'"
                )
            seen_aliases[alias] = key

    _TEMPLATE_FIELDS = (
        "round1_header", "round1_instruction", "round1_structure",
        "round2_instruction", "synthesis_output",
    )
    _REQUIRED_FIELDS = ("round1_instruction", "synthesis_output")
    for key, cfg in modes.items():
        has_any_template = any(getattr(cfg, f).strip() for f in _TEMPLATE_FIELDS)
        if has_any_template:
            missing = [f for f in _REQUIRED_FIELDS if not getattr(cfg, f).strip()]
            if missing:
                raise ValueError(
                    f"Mode '{key}' missing required template fields: {missing}"
                )


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
            cost_per_1m_input=float(model_raw.get("cost_per_1m_input", 0.0)),
            cost_per_1m_output=float(model_raw.get("cost_per_1m_output", 0.0)),
        )
        models[provider_name] = model_cfg

        api_key = os.environ.get(model_raw["api_key_env"], "").strip()
        if api_key:
            available_providers.add(provider_name)
            logger.info("Provider available: %s", provider_name)
        else:
            logger.info(
                "Provider skipped (no API key): %s — set %s in global env",
                provider_name,
                model_raw["api_key_env"],
            )

    inbox_raw = raw.get("inbox", {})
    inbox = InboxConfig(
        dir=Path(inbox_raw.get("dir", "./council_inbox")),
        archive_dir=Path(inbox_raw.get("archive_dir", "./council_inbox/archive")),
        downloads_dir=Path(inbox_raw.get("downloads_dir", "~/Downloads")).expanduser(),
        scan_downloads=bool(inbox_raw.get("scan_downloads", True)),
        council_frontmatter_keys=list(
            inbox_raw.get("council_frontmatter_keys", ["mode", "rounds", "models", "synthesizer", "full"])
        ),
    )

    # Parse modes (optional — falls back to empty dict if section absent)
    modes: dict[str, ModeConfig] = {}
    for mode_key, mode_raw in raw.get("modes", {}).items():
        modes[mode_key] = ModeConfig(
            description=str(mode_raw.get("description", "")),
            emoji=str(mode_raw.get("emoji", "")),
            aliases=list(mode_raw.get("aliases", [])),
            default=bool(mode_raw.get("default", False)),
            max_rounds=int(mode_raw.get("max_rounds", 2)),
            token_budget=int(mode_raw.get("token_budget", 1500)),
            round1_header=str(mode_raw.get("round1_header", "")),
            round1_instruction=str(mode_raw.get("round1_instruction", "")),
            round1_structure=str(mode_raw.get("round1_structure", "")),
            round2_instruction=str(mode_raw.get("round2_instruction", "")),
            synthesis_output=str(mode_raw.get("synthesis_output", "")),
        )

    if modes:
        _validate_modes(modes)

    # Parse per-model persona directives: {mode: {provider: directive}}
    persona_mode_directives: dict[str, dict[str, str]] = {}
    for mode_key, provider_map in raw.get("persona_mode_directives", {}).items():
        persona_mode_directives[mode_key] = {
            k: str(v) for k, v in (provider_map or {}).items()
        }

    # Parse research config (optional section)
    research: ResearchConfig | None = None
    if "research" in raw:
        research = _load_research_config(raw["research"])

    return AppConfig(
        defaults=defaults,
        models=models,
        prompts=prompts,
        inbox=inbox,
        available_providers=available_providers,
        modes=modes,
        persona_mode_directives=persona_mode_directives,
        research=research,
    )


def _load_research_config(raw: dict) -> ResearchConfig:
    """Parse the research: section of settings.yaml."""
    providers: dict[str, ResearchProviderConfig] = {}
    for provider_name, p_raw in raw.get("providers", {}).items():
        providers[provider_name] = ResearchProviderConfig(
            name=provider_name,
            model=str(p_raw["model"]),
            api_key_env=str(p_raw["api_key_env"]),
            timeout_sec=int(p_raw.get("timeout_sec", 60)),
            cost_per_1m_input=float(p_raw.get("cost_per_1m_input", 0.0)),
            cost_per_1m_output=float(p_raw.get("cost_per_1m_output", 0.0)),
            base_url=str(p_raw["base_url"]) if "base_url" in p_raw else None,
        )

    cache_dir_raw = str(raw.get("cache_dir", "~/.ai-council/research_cache"))
    cache_dir = Path(cache_dir_raw).expanduser()

    return ResearchConfig(
        default_providers=list(raw.get("default_providers", [])),
        deep_providers=list(raw.get("deep_providers", [])),
        cache_dir=cache_dir,
        cache_ttl_days=int(raw.get("cache_ttl_days", 7)),
        summary_max_tokens=int(raw.get("summary_max_tokens", 2500)),
        summary_model=str(raw.get("summary_model", "deepseek")),
        providers=providers,
    )
