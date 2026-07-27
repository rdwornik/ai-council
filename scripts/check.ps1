#!/usr/bin/env pwsh
# check.ps1 — CI-equivalent check for solo dev. Run before merging any branch.
#
# Every tool runs through the REPO VENV, explicitly (#123). Bare `pytest` / `mypy` / `ruff` /
# `python` resolve to whatever is first on PATH, and on this machine that is the system
# interpreter, which carries its own editable `ai_council` install — so `import ai_council`
# resolved to the PRIMARY checkout from any working directory, and a worktree silently tested
# the primary's code (the defect `conftest.py` now guards). Naming the interpreter removes the
# ambiguity at the source instead of relying on an activated shell.

$ErrorActionPreference = "Stop"
# Native non-zero exits must set $LASTEXITCODE, NOT throw. With this preference enabled (it is
# session-configurable, and the default has moved before) `ErrorActionPreference = "Stop"`
# promotes any non-zero native exit to a terminating error -- which would abort the gate on the
# claim checker's exit 1 and quietly break its non-blocking contract. Pinned here rather than
# inherited, so the gate behaves the same in every operator's shell (terra pre-merge, #123).
$PSNativeCommandUseErrorActionPreference = $false

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $RepoRoot ".venv\Scripts\python.exe"

# Fail LOUD if the venv is absent rather than falling back to a bare runner: a silent fallback
# is exactly the ambiguity this change removes, and it would look like a normal green run.
if (-not (Test-Path $Py)) {
    Write-Host "check.ps1: repo venv not found at $Py" -ForegroundColor Red
    Write-Host "  The gate runs through the repo venv on purpose (#123) and will not fall back" -ForegroundColor Red
    Write-Host "  to a bare interpreter, because that is how a run against the wrong tree looks" -ForegroundColor Red
    Write-Host "  identical to a correct one. Create it:" -ForegroundColor Red
    Write-Host "      py -m venv .venv" -ForegroundColor Yellow
    Write-Host "      .venv\Scripts\python.exe -m pip install -e `".[dev]`"" -ForegroundColor Yellow
    exit 1
}

Write-Host "Interpreter: $Py" -ForegroundColor DarkGray
Write-Host "Repo root  : $RepoRoot" -ForegroundColor DarkGray

# Run from the repo root, not the caller's cwd. Pinning only the INTERPRETER was not enough:
# `tests/`, `src/` and the checker path are relative, so invoking a worktree's check.ps1 from
# another checkout would use the worktree's venv while testing the OTHER tree's files -- the
# same false-green this ticket exists to close, just relocated (terra pre-merge, #123).
Push-Location $RepoRoot
try {

Write-Host "Running pytest..." -ForegroundColor Cyan
& $Py -m pytest tests/ -m "not integration and not envcheck" -v
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Running mypy..." -ForegroundColor Cyan
& $Py -m mypy src/
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Running ruff..." -ForegroundColor Cyan
& $Py -m ruff check src/ tests/
if ($LASTEXITCODE -ne 0) { exit 1 }

# Non-blocking claim-vs-reality report (#97). Exit is captured for visibility but deliberately
# NOT propagated -- the gate stays pytest+mypy+ruff. A checker crash (exit >=2) is surfaced in
# Red so it can never be mistaken for a clean pass; a promotion to gating is a one-line change.
Write-Host "Running claim-vs-reality checker (non-blocking)..." -ForegroundColor Yellow
& $Py scripts/validate_claims.py
$claimsExit = $LASTEXITCODE
if ($claimsExit -ge 2) {
    Write-Host "claim checker ERRORED (exit $claimsExit) - findings unreliable this run" -ForegroundColor Red
}

Write-Host "All checks passed!" -ForegroundColor Green
# The gate passed (any gate failure exited 1 above). Reset the exit code so the non-blocking
# checker's exit (1 = findings) does not leak into check.ps1's result. To PROMOTE the checker
# to gating, replace this line with: if ($claimsExit -ge 1) { exit 1 } else { exit 0 }
exit 0

}
finally {
    # Always restore the caller's location, including on the `exit 1` paths above.
    Pop-Location
}
