"""Panel and provider utility functions.

CouncilRunner (the debate orchestrator) lives in src/orchestrator.py.
This module re-exports it for backward compatibility.
"""

import logging

from ai_council.providers.base import AIProvider
from config.config_loader import AppConfig

logger = logging.getLogger(__name__)


def build_all_providers(config: AppConfig, provider_classes: dict) -> dict[str, AIProvider]:
    """Instantiate all available providers from config. Returns dict keyed by name."""
    providers: dict[str, AIProvider] = {}
    for name in config.available_providers:
        if name not in provider_classes:
            logger.warning("Provider '%s' unknown, skipping", name)
            continue
        model_cfg = config.models[name]
        try:
            providers[name] = provider_classes[name](model_cfg)
        except Exception as exc:
            logger.warning("Failed to instantiate provider '%s': %s", name, exc)
    return providers


def determine_panel(
    config: AppConfig,
    models_arg: str | None,
    full_flag: bool,
) -> tuple[list[str], str]:
    """Returns (panel_names, panel_mode). --models wins over --full wins over default."""
    if models_arg:
        return [m.strip() for m in models_arg.split(",")], "custom"
    elif full_flag:
        return config.defaults.full_panel, "full"
    else:
        return config.defaults.default_panel, "default"


def exclude_synthesizer_from_panel(
    panel_names: list[str],
    synthesizer_name: str,
    all_providers: dict[str, AIProvider],
) -> list[str]:
    """Remove synthesizer from panel when doing so still leaves >= 2 available debaters."""
    if synthesizer_name not in panel_names:
        return panel_names
    remaining = [n for n in panel_names if n != synthesizer_name]
    available_remaining = [n for n in remaining if n in all_providers]
    if len(available_remaining) >= 2:
        return remaining
    return panel_names


def pick_synthesizer(
    all_providers: dict[str, AIProvider],
    panel_names: list[str],
    preferred: str,
) -> tuple[AIProvider, bool]:
    """Pick synthesizer not in panel. Returns (provider, is_participant).

    is_participant=True only when no non-participant is available.
    """
    not_in_panel = [n for n in all_providers if n not in panel_names]
    if not_in_panel:
        if preferred in not_in_panel:
            return all_providers[preferred], False
        return all_providers[not_in_panel[0]], False
    if preferred in all_providers:
        return all_providers[preferred], True
    return next(iter(all_providers.values())), True


# Backward-compat re-export — new code should import from ai_council.orchestrator directly
from ai_council.orchestrator import CouncilRunner as CouncilRunner  # noqa: E402, F401
