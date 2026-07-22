"""Load settings.yaml into typed dataclasses. Validates API keys at startup."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_SETTINGS_PATH = Path(__file__).parent / "settings.yaml"
_REPO_ROOT = Path(__file__).parent.parent


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
    # ADR-12 backend axis. Default "api" everywhere — the §5 flip to CLI is evidence-gated
    # on CLI-4 parity (#27), not enabled here. When backend == "cli": cli_command drives the
    # subprocess (claude|codex) with cli_model pinned per call; the API `model` above is the
    # same-seat API fallback target.
    backend: str = "api"             # "api" | "cli"
    cli_command: str | None = None   # "claude" | "codex" when backend == "cli"
    cli_model: str | None = None     # model pinned on every CLI call (per-call pin rule)


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
        default_factory=lambda: ["mode", "rounds", "models", "synthesizer", "full", "target-project"]
    )


@dataclass
class DefaultsConfig:
    rounds: int
    max_rounds: int
    output_dir: Path
    synthesizer: str
    default_panel: list[str] = field(default_factory=list)
    full_panel: list[str] = field(default_factory=list)
    secondary_output_dir: Path | None = None
    secondary_output_enabled: bool = True


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
    poll_interval_sec: int = 10
    reasoning_effort: str | None = None  # e.g. "low", "medium", "high" — for reasoning-capable models


@dataclass
class ResearchConfig:
    default_providers: list[str]
    deep_providers: list[str]
    cache_dir: Path
    cache_ttl_days: int
    summary_max_tokens: int
    summary_model: str
    providers: dict[str, ResearchProviderConfig] = field(default_factory=dict)
    # Denominator = selected panel post --models filter; build-time dropouts count as failures.
    # Run completes but exits with code 3 (see ADR-08).
    min_successful_providers: int = 3


@dataclass
class CruxCheckConfig:
    """The bounded between-rounds crux check (#18).

    ``providers`` is a DELIBERATELY NARROW subset of ``research.providers`` — the step is
    unconditional, so it runs on EVERY debate. The full research panel carries 240-1800s
    timeouts and would add up to 30 minutes to a trivial pick run; ``budget_sec`` is the
    hard wall-clock cap that keeps it affordable.

    There is no ``model`` key: the extraction call reuses the already-selected synthesizer
    instance (a non-participant by default), so no panelist gains an asymmetric role.
    """

    providers: list[str] = field(default_factory=list)
    budget_sec: float = 90.0
    injection_header: str = ""
    extraction_prompt: str = ""


@dataclass
class BoostConfig:
    """The `council boost` input stage (Unit 2 P1, ADR-11 boost→decide chain).

    Both prompts are cheap single calls in the detect_mode shape. Their output is
    ADVISORY (hybrid gate posture): classification is validated by a deterministic
    heuristic, and the decompose answer must pass boost.py's hard verbatim gate.
    """

    classify_prompt: str
    decompose_prompt: str
    timeout_sec: float = 20.0


@dataclass
class AppConfig:
    defaults: DefaultsConfig
    models: dict[str, ModelConfig]
    prompts: PromptsConfig
    inbox: InboxConfig = field(
        default_factory=lambda: InboxConfig(
            Path("./council_inbox"), Path("./council_inbox/archive")
        )
    )
    available_providers: set[str] = field(default_factory=set)
    modes: dict[str, ModeConfig] = field(default_factory=dict)
    persona_mode_directives: dict[str, dict[str, str]] = field(default_factory=dict)
    research: ResearchConfig | None = None
    # #18 bounded crux check; None when the section is absent → the step is simply not built.
    crux_check: CruxCheckConfig | None = None
    # Unit 2 P1 boost stage; None when the section is absent → `council boost` fails loud.
    boost: BoostConfig | None = None
    dev_root: Path | None = None
    target_projects: list[str] = field(default_factory=list)
    # Raw RunPolicy `policy:` block (B7); None when absent → RunPolicy code defaults apply.
    policy: dict[str, int] | None = None


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
    # Resolve output_dir relative to repo root so transcripts land in the same
    # place regardless of which directory the user runs `council` from.
    output_dir = (_REPO_ROOT / defaults_raw["output_dir"]).resolve()

    secondary_dir_raw = defaults_raw.get("secondary_output_dir")
    secondary_dir = Path(secondary_dir_raw).expanduser() if secondary_dir_raw else None
    secondary_enabled = bool(defaults_raw.get("secondary_output_enabled", True))

    defaults = DefaultsConfig(
        rounds=int(defaults_raw["rounds"]),
        max_rounds=int(defaults_raw["max_rounds"]),
        output_dir=output_dir,
        synthesizer=str(defaults_raw["synthesizer"]),
        default_panel=list(defaults_raw["default_panel"]),
        full_panel=list(defaults_raw["full_panel"]),
        secondary_output_dir=secondary_dir,
        secondary_output_enabled=secondary_enabled,
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
            backend=str(model_raw.get("backend", "api")),
            cli_command=str(model_raw["cli_command"]) if "cli_command" in model_raw else None,
            cli_model=str(model_raw["cli_model"]) if "cli_model" in model_raw else None,
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
        dir=(_REPO_ROOT / inbox_raw.get("dir", "council_inbox")).resolve(),
        archive_dir=(_REPO_ROOT / inbox_raw.get("archive_dir", "council_inbox/archive")).resolve(),
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

    # Parse crux_check config (optional section; absent → the step is not built)
    crux_check: CruxCheckConfig | None = None
    if "crux_check" in raw:
        crux_check = _load_crux_check_config(raw["crux_check"] or {})

    # Parse boost config (optional section; absent → `council boost` fails loud)
    boost: BoostConfig | None = None
    if "boost" in raw:
        boost = _load_boost_config(raw["boost"] or {})

    # Parse target_projects (new schema per ADR-43 amendment cycle 1, 2026-05-11):
    # dev_root + list of project names; paths computed at resolve time.
    raw_tp = raw.get("target_projects", [])

    if isinstance(raw_tp, dict):
        raise ValueError(
            "target_projects schema changed 2026-05-11 (ADR-43 amendment cycle 1). "
            "Expected: list of project names. Found: dict. "
            "See README.md Transcript Routing section."
        )
    if not isinstance(raw_tp, list):
        raise ValueError(
            f"target_projects must be a list of project name strings, got {type(raw_tp).__name__}"
        )
    for i, item in enumerate(raw_tp):
        if not isinstance(item, str):
            raise ValueError(
                f"target_projects items must be strings; got {type(item).__name__!r} at index {i}: {item!r}"
            )
    seen: set[str] = set()
    for name in raw_tp:
        if name in seen:
            raise ValueError(f"Duplicate project name in target_projects: {name!r}")
        seen.add(name)
    target_projects: list[str] = list(raw_tp)

    # Parse dev_root — required when target_projects is non-empty
    dev_root: Path | None = None
    if target_projects:
        raw_dev_root = raw.get("dev_root")
        if raw_dev_root is None:
            raise ValueError(
                "dev_root is required in settings.yaml when target_projects is non-empty"
            )
        if not isinstance(raw_dev_root, str):
            raise ValueError(
                f"dev_root must be a string, got {type(raw_dev_root).__name__!r}"
            )
        dev_root_path = Path(raw_dev_root).expanduser().resolve()
        if not dev_root_path.is_dir():
            raise ValueError(
                f"dev_root must point to existing directory: {dev_root_path}"
            )
        dev_root = dev_root_path

    # Parse the RunPolicy block (optional). Coerce to int so RunPolicy.from_config
    # gets a clean dict[str, int]; absent block → None → RunPolicy code defaults.
    policy_raw = raw.get("policy")
    policy: dict[str, int] | None = (
        {k: int(v) for k, v in policy_raw.items()} if policy_raw else None
    )

    return AppConfig(
        defaults=defaults,
        models=models,
        prompts=prompts,
        inbox=inbox,
        available_providers=available_providers,
        modes=modes,
        persona_mode_directives=persona_mode_directives,
        research=research,
        crux_check=crux_check,
        boost=boost,
        dev_root=dev_root,
        target_projects=target_projects,
        policy=policy,
    )


def _load_crux_check_config(raw: dict) -> CruxCheckConfig:
    """Parse the crux_check: section of settings.yaml (#18)."""
    return CruxCheckConfig(
        providers=list(raw.get("providers", [])),
        budget_sec=float(raw.get("budget_sec", 90.0)),
        injection_header=str(raw.get("injection_header", "")),
        extraction_prompt=str(raw.get("extraction_prompt", "")),
    )


def _load_boost_config(raw: dict) -> BoostConfig:
    """Parse the boost: section of settings.yaml (Unit 2 P1). Fails loud on a
    missing prompt — a boost section without its prompts is a config error, not
    a silent heuristic-only downgrade."""
    missing = [k for k in ("classify_prompt", "decompose_prompt") if not str(raw.get(k, "")).strip()]
    if missing:
        raise ValueError(f"boost: section is missing required prompt(s): {missing}")
    return BoostConfig(
        classify_prompt=str(raw["classify_prompt"]),
        decompose_prompt=str(raw["decompose_prompt"]),
        timeout_sec=float(raw.get("timeout_sec", 20.0)),
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
            poll_interval_sec=int(p_raw.get("poll_interval_sec", 10)),
            reasoning_effort=(
                str(p_raw["reasoning_effort"]) if "reasoning_effort" in p_raw else None
            ),
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
        min_successful_providers=int(raw.get("min_successful_providers", 3)),
    )
