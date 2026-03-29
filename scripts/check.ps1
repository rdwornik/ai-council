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

Write-Host "All checks passed!" -ForegroundColor Green
