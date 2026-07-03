---
name: override
description: Bypass the ADR-85 session-end hard gate for this HEAD — explicit, logged, no auto-bypass
---

The session-end Stop-hook (`scripts/session_end_backpressure.py`, ADR-85) **hard-blocks**
turn-end when commits are ahead of base but the `JOURNAL.md` entry names no commit-SHA from
this session. That block is *fail-closed but overridable* — and `/override [reason]` is the
**only** exit. There is deliberately no auto-bypass-after-cap (that would train "persistence
beats policy"). Use this only when the block is genuinely wrong (e.g. a deterministic
false-positive), not to skip writing the journal.

`$ARGUMENTS` is the reason.

## Procedure

1. **Require a reason.** If `$ARGUMENTS` is empty or whitespace → **reject**: do nothing,
   write nothing, and tell the operator a reason is mandatory. Never arm an unexplained
   override.
2. **Log + arm**, in this order (the token must record the HEAD the Stop-hook will see, so
   do **not** commit anything in between — both files are gitignored, so the tree stays
   clean and HEAD does not move):
   - Append a newest-first entry to `logs/OVERRIDES.md` (gitignored ephemeral local audit):
     timestamp, reason, branch, HEAD.
   - Write the HEAD-bound one-shot token `logs/.session-override-token` (gitignored) =
     `{ "head": <current HEAD>, "reason": <reason>, "ts": <timestamp> }`.

   Run this (PowerShell):

   ```powershell
   $reason = "$ARGUMENTS".Trim()
   if (-not $reason) { Write-Output "REJECTED — /override requires a reason. Nothing armed."; return }
   $head   = (git rev-parse HEAD).Trim()
   $branch = (git branch --show-current).Trim()
   $ts     = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
   if (-not (Test-Path logs)) { New-Item -ItemType Directory logs | Out-Null }  # logs/ is not tracked (gitignored-ephemeral); create on demand
   if (-not (Test-Path logs/OVERRIDES.md)) { Set-Content logs/OVERRIDES.md "# Session-gate overrides (ADR-85)`n`nAppend-only, newest-first. Gitignored ephemeral local audit (override-rate telemetry).`n" -Encoding utf8 }
   $existing = Get-Content logs/OVERRIDES.md -Raw
   $entry = "## $ts — $branch @ $($head.Substring(0,7))`n$reason`n`n"
   # prepend the new entry directly under the H1 header block (newest-first)
   $parts = $existing -split "(?<=telemetry\)\.`r?`n)", 2
   Set-Content logs/OVERRIDES.md ($parts[0] + "`n" + $entry + $parts[1]) -Encoding utf8 -NoNewline
   @{ head = $head; reason = $reason; ts = $ts } | ConvertTo-Json -Compress | Set-Content logs/.session-override-token -Encoding utf8 -NoNewline
   Write-Output "OVERRIDE ARMED — HEAD $($head.Substring(0,7)) on $branch. The next turn-end passes once for this HEAD; it re-arms automatically when a new commit lands."
   ```

3. **Report** what was logged and that the gate will allow turn-end for the current HEAD.

## Notes
- **HEAD-bound, one-shot.** The token allows the gate only while HEAD is unchanged. The
  moment a new commit lands, HEAD moves and the token is stale → the gate re-arms. The
  Stop-hook only *reads* the token (it stays a read-only validator — Critical Rule #4).
- **Never committed.** `logs/.session-override-token` and `logs/OVERRIDES.md` are gitignored
  (ADR-85). Arming an override does not dirty the tracked tree and creates no leftover.
- **Telemetry.** `logs/OVERRIDES.md` is the local override-rate record — if overrides exceed
  ~10% of sessions, the gate rules need tuning, not the human reinstated as trigger (ADR-85
  consequences).
