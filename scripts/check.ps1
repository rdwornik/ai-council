#!/usr/bin/env pwsh
# check.ps1 — CI-equivalent check for solo dev. Run before merging any branch.

Write-Host "Running pytest..." -ForegroundColor Cyan
pytest tests/ -m "not integration and not envcheck" -v
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Running mypy..." -ForegroundColor Cyan
mypy src/
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Running ruff..." -ForegroundColor Cyan
ruff check src/ tests/
if ($LASTEXITCODE -ne 0) { exit 1 }

# Non-blocking claim-vs-reality report (#97). Exit is captured for visibility but deliberately
# NOT propagated -- the gate stays pytest+mypy+ruff. A checker crash (exit >=2) is surfaced in
# Red so it can never be mistaken for a clean pass; a promotion to gating is a one-line change.
Write-Host "Running claim-vs-reality checker (non-blocking)..." -ForegroundColor Yellow
python scripts/validate_claims.py
$claimsExit = $LASTEXITCODE
if ($claimsExit -ge 2) {
    Write-Host "claim checker ERRORED (exit $claimsExit) - findings unreliable this run" -ForegroundColor Red
}

Write-Host "All checks passed!" -ForegroundColor Green
# The gate passed (any gate failure exited 1 above). Reset the exit code so the non-blocking
# checker's exit (1 = findings) does not leak into check.ps1's result. To PROMOTE the checker
# to gating, replace this line with: if ($claimsExit -ge 1) { exit 1 } else { exit 0 }
exit 0
