"""council doctor -- liveness + config pre-flight (DRAFT-DOC-1 v1).

Advisory only: prints a one-screen GREEN/YELLOW/RED truth table over
KEYS -> SEATS -> CONFIG and writes a versioned machine-readable verdict record
to ``<output_dir>/health/``. It NEVER blocks a subsequent council run -- enforcement
stays with the run-time health gate (cli.py) and Lane-A caller obligations.

v1 scope is liveness + static config validation. Deferred to follow-on arcs:
pin-currency sweep (DRAFT-DOC-2), CLI-fleet / identity-channel --smoke re-probe
(the L-CLI seam contribution), and advisory first_seen aging.

Consumes ``healthcheck.run_health_checks`` (do-not-touch reference module) for the
seat pings; it does not reimplement provider health-checking.
"""

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.console import Console

from ai_council.healthcheck import run_health_checks
from ai_council.providers.base import AIProvider
from ai_council.runner import build_all_providers
from config.config_loader import AppConfig

# Committed surface (§C.2): version from day one. Field removals/renames are breaking
# changes once a foreign repo consumes doctor-latest.json (ADR-11 §5 pattern).
SCHEMA_VERSION = "1"

GREEN, YELLOW, RED = "GREEN", "YELLOW", "RED"
PASS, ADVISORY, FAIL = "PASS", "ADVISORY", "FAIL"

# Exit codes mirror the verdict in ADR-08 spirit (YELLOW=3 == research-degradation code).
_EXIT = {GREEN: 0, YELLOW: 3, RED: 1}

# ASCII-only status marks (Windows cp1252 anti-pattern: no box-drawing / non-ASCII glyphs).
_MARK = {PASS: "[green]OK [/green]", ADVISORY: "[yellow] ! [/yellow]", FAIL: "[red] X [/red]"}
_VERDICT_COLOR = {GREEN: "[bold green]GREEN[/bold green]",
                  YELLOW: "[bold yellow]YELLOW[/bold yellow]",
                  RED: "[bold red]RED[/bold red]"}


@dataclass
class Check:
    """One doctor finding. ``subject`` is a key-env NAME / seat name / config ref --
    never a secret value. ``role`` labels seat rows (debate | synthesizer)."""

    category: str   # "key" | "seat" | "config"
    subject: str
    status: str     # PASS | ADVISORY | FAIL
    detail: str
    role: str | None = None


def check_keys(config: AppConfig, shell_snapshot: dict[str, str | None]) -> list[Check]:
    """KEYS row: each configured api_key_env present & non-empty. NAME only, never values.

    A key that was set-but-empty in the launching shell (env shadowing) is reported as a
    non-fatal ADVISORY -- the doctor loaded global secrets with override, so the real value
    is in use, but the poisoned shell is named.
    """
    env_to_seats: dict[str, list[str]] = {}
    for name, model in config.models.items():
        env_to_seats.setdefault(model.api_key_env, []).append(name)

    checks: list[Check] = []
    for env, seats in sorted(env_to_seats.items()):
        value = os.environ.get(env, "").strip()
        seats_str = ", ".join(sorted(seats))
        if value:
            if shell_snapshot.get(env) == "":
                checks.append(Check(
                    "key", env, ADVISORY,
                    f"set-but-empty in shell; global secrets used ({seats_str})",
                ))
            else:
                checks.append(Check("key", env, PASS, f"present ({seats_str})"))
        else:
            checks.append(Check("key", env, FAIL, f"absent -- {seats_str} cannot run"))
    return checks


def check_seats(config: AppConfig, provider_classes: dict[str, type[AIProvider]]) -> list[Check]:
    """SEATS row: live liveness ping of every buildable provider via run_health_checks.

    The synthesizer seat is reported on its own row (never folded into the debate roll-up).
    A build/ping failure is contained to its own FAIL row -- it never crashes the doctor.
    """
    try:
        providers = build_all_providers(config, provider_classes)
    except Exception as exc:  # containment: a build blow-up is a row, not a crash
        return [Check("seat", "(build)", FAIL, f"provider build failed: {exc}")]

    if not providers:
        return [Check("seat", "(all)", FAIL, "no providers built -- check API keys")]

    try:
        results = run_health_checks_sync(providers)
    except Exception as exc:  # containment
        return [Check("seat", "(pings)", FAIL, f"health check errored: {exc}")]

    synthesizer = config.defaults.synthesizer
    checks: list[Check] = []
    for name, (ok, err) in sorted(results.items()):
        role = "synthesizer" if name == synthesizer else "debate"
        if ok:
            checks.append(Check("seat", name, PASS, "ping OK", role=role))
        else:
            checks.append(Check("seat", name, FAIL, err or "ping failed", role=role))
    return checks


def run_health_checks_sync(providers: dict[str, AIProvider]) -> dict[str, tuple[bool, str]]:
    """Thin sync wrapper so callers/tests can patch a single seam over the async ping."""
    return asyncio.run(run_health_checks(providers))


def validate_config(config: AppConfig) -> list[Check]:
    """CONFIG row: pure static reference resolution -- zero LLM spend, zero network.

    Confirms the synthesizer + panel names resolve to models{}, and the research
    summary/provider names resolve to research.providers{}, plus a min-successful sanity.
    """
    checks: list[Check] = []
    model_names = set(config.models)

    synth = config.defaults.synthesizer
    if synth in model_names:
        checks.append(Check("config", "synthesizer", PASS, f"'{synth}' resolves to models"))
    else:
        checks.append(Check("config", "synthesizer", FAIL, f"'{synth}' not in models"))

    for label, panel in (("default_panel", config.defaults.default_panel),
                         ("full_panel", config.defaults.full_panel)):
        missing = [p for p in panel if p not in model_names]
        if missing:
            checks.append(Check("config", label, FAIL, f"unresolved: {', '.join(missing)}"))
        else:
            checks.append(Check("config", label, PASS, f"{len(panel)} seats resolve"))

    research = config.research
    if research is not None:
        research_names = set(research.providers)

        # summary_model resolves against the TOP-LEVEL models pool, not research.providers
        # (merger.py + cli.py _check_summarizer_health). An unresolved name is not run-fatal:
        # merger.py degrades to a truncation fallback -- so it is an ADVISORY, not a FAIL.
        summary = research.summary_model
        if summary in model_names:
            checks.append(Check("config", "research.summary_model", PASS,
                                f"'{summary}' resolves to models"))
        else:
            checks.append(Check("config", "research.summary_model", ADVISORY,
                                f"'{summary}' not in models -- research uses truncation fallback"))

        # default/deep provider rosters resolve against research.providers.
        for label, names in (("research.default_providers", research.default_providers),
                             ("research.deep_providers", research.deep_providers)):
            missing = [p for p in names if p not in research_names]
            if missing:
                checks.append(Check("config", label, FAIL, f"unresolved: {', '.join(missing)}"))
            else:
                checks.append(Check("config", label, PASS, f"{len(names)} resolve"))

        n_default = len(research.default_providers)
        if n_default and research.min_successful_providers > n_default:
            checks.append(Check(
                "config", "research.min_successful_providers", ADVISORY,
                f"{research.min_successful_providers} > {n_default} default providers (unsatisfiable)",
            ))
        else:
            checks.append(Check(
                "config", "research.min_successful_providers", PASS,
                f"{research.min_successful_providers} <= {n_default} default providers",
            ))
    return checks


def _collect_secret_values(config: AppConfig) -> list[str]:
    """Every non-empty credential VALUE currently in the environment, for redaction.
    Length is not filtered -- under-redaction leaks, over-redaction is merely cosmetic,
    so any non-empty configured credential is redacted regardless of length."""
    envs = {m.api_key_env for m in config.models.values()}
    if config.research is not None:
        envs |= {p.api_key_env for p in config.research.providers.values()}
    values = []
    for env in envs:
        value = os.environ.get(env, "")
        if value:
            values.append(value)
    return values


def _redact(text: str, secrets: list[str]) -> str:
    """Defense-in-depth: strip any raw credential value that a provider exception or
    health-check string might carry, before it reaches the screen or the JSON record.
    The contract is keys by NAME only -- values never appear (DRAFT-DOC-1)."""
    for secret in secrets:
        if secret and secret in text:
            text = text.replace(secret, "[REDACTED]")
    return text


def evaluate_verdict(checks: list[Check]) -> str:
    """Any FAIL -> RED; else any ADVISORY -> YELLOW; else GREEN."""
    if any(c.status == FAIL for c in checks):
        return RED
    if any(c.status == ADVISORY for c in checks):
        return YELLOW
    return GREEN


def build_record(checks: list[Check], verdict: str, generated_at: str) -> dict:
    """The machine-readable §C.2 record. Keys appear by NAME only; no secret values."""
    seats: dict[str, dict[str, str]] = {}
    for c in checks:
        if c.category == "seat" and c.role:
            seats.setdefault(c.role, {})[c.subject] = c.status
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "verdict": verdict,
        "checks": [
            {"class": c.category, "subject": c.subject, "status": c.status, "detail": c.detail}
            for c in checks
        ],
        "seats": seats,
    }


def write_record(record: dict, output_dir: Path, filestamp: str) -> Path:
    """Write ``doctor-<ts>.json`` + rewrite ``doctor-latest.json`` (a full copy, not a
    symlink -- Windows). Returns the timestamped record path."""
    health_dir = Path(output_dir) / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, indent=2) + "\n"

    record_path = health_dir / f"doctor-{filestamp}.json"
    record_path.write_text(payload, encoding="utf-8")
    (health_dir / "doctor-latest.json").write_text(payload, encoding="utf-8")
    return record_path


def render_report(console: Console, checks: list[Check], verdict: str,
                  record_path: Path | None, generated_at: str) -> None:
    """One-screen ASCII report, fixed epistemic-load order KEYS -> SEATS -> CONFIG."""
    console.print(f"\n[bold]COUNCIL DOCTOR[/bold] -- {generated_at} -- verdict: {_VERDICT_COLOR[verdict]}\n")

    console.print("[bold]KEYS[/bold]")
    for c in (c for c in checks if c.category == "key"):
        console.print(f"  {_MARK[c.status]} {c.subject}: {c.detail}")

    console.print("[bold]SEATS[/bold]")
    seat_checks = [c for c in checks if c.category == "seat"]
    for c in (c for c in seat_checks if c.role != "synthesizer"):
        console.print(f"  {_MARK[c.status]} {c.subject}: {c.detail}")
    for c in (c for c in seat_checks if c.role == "synthesizer"):
        console.print(f"  {_MARK[c.status]} {c.subject} (synthesizer): {c.detail}")

    console.print("[bold]CONFIG[/bold]")
    for c in (c for c in checks if c.category == "config"):
        console.print(f"  {_MARK[c.status]} {c.subject}: {c.detail}")

    console.print(f"\nverdict: {_VERDICT_COLOR[verdict]}  (doctor never blocks a run)")
    console.print(f"record: {record_path if record_path is not None else '(not written)'}\n")


def run_doctor(
    config: AppConfig,
    provider_classes: dict[str, type[AIProvider]],
    *,
    shell_snapshot: dict[str, str | None] | None = None,
    console: Console | None = None,
    output_dir: Path | None = None,
) -> int:
    """Run the v1 doctor end-to-end. Returns the exit code (0=GREEN / 3=YELLOW / 1=RED).

    ``shell_snapshot`` maps each key-env NAME to its raw value in the launching shell
    (before the doctor's override-load) so env shadowing can be surfaced as an advisory.
    """
    console = console or Console(legacy_windows=False)
    shell_snapshot = shell_snapshot or {}
    now = datetime.now()
    generated_at = now.isoformat(timespec="seconds")
    filestamp = now.strftime("%Y-%m-%dT%H%M%S")

    checks: list[Check] = []
    checks += check_keys(config, shell_snapshot)
    checks += check_seats(config, provider_classes)
    checks += validate_config(config)

    # Redact any raw credential value a provider exception / ping string may carry, so
    # values never reach the screen or the persisted record (keys by NAME only).
    secrets = _collect_secret_values(config)
    if secrets:
        checks = [
            Check(c.category, _redact(c.subject, secrets), c.status, _redact(c.detail, secrets), c.role)
            for c in checks
        ]

    verdict = evaluate_verdict(checks)
    out_dir = output_dir if output_dir is not None else config.defaults.output_dir
    record = build_record(checks, verdict, generated_at)

    # The record is a best-effort side artifact. A filesystem failure must not crash the
    # doctor (containment) nor flip the health verdict -- a GREEN council whose record
    # could not be written is still GREEN; a Lane-A caller must not be told "don't
    # commission" over a local write error. Warn, keep the verdict, carry on.
    try:
        record_path: Path | None = write_record(record, out_dir, filestamp)
    except OSError as exc:
        record_path = None
        console.print(f"[yellow]WARNING:[/yellow] could not write doctor record to {out_dir}: {exc}")

    render_report(console, checks, verdict, record_path, generated_at)
    return _EXIT[verdict]
