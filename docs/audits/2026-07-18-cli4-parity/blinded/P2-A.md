# AI Council Debate: What mechanism should host the recurring local cross-repo fleet-baseline run?

**Panel:** claude-opus-4-8, gpt-5.6-terra
**Synthesizer:** gemini (non-participant)
**Rounds:** 2
**Panel Mode:** custom
**Debate Mode:** pick
**Source:** cli

## Verdict Summary

**Decision:** Host the run using Windows Task Scheduler to invoke a version-controlled PowerShell script, conditionally.
**Dissent:** non-unanimous (see minority report)
**Panel seated:** 2/2
**Verdict author:** gemini (non-participant)

_Machine-readable fields are authoritative in the council-verdict-*.json sibling._

---

## Question

What mechanism should host the recurring local cross-repo fleet-baseline run?

## Round 1: Initial Responses

### Claude (claude-opus-4-8)

## Position
Host the recurring run on **Windows Task Scheduler invoking a single, version-controlled, idempotent PowerShell orchestrator script** — the scheduler owns *when*, the checked-in script owns *what*. Do not embed cross-repo logic in the scheduler's arguments or in CI.

## Alternatives Considered
- **CI-hosted (GitHub Actions scheduled workflow):** Rejected as the *primary* host because the run is explicitly "local" and "cross-repo." CI runners are ephemeral, clone one repo by default, and don't have your local fleet layout, credentials, or OneDrive-excluded paths. You'd fight the sandbox to reconstruct local state. It's a fine *mirror/verifier*, not the owner.
- **WSL cron / systemd timer:** Rejected. Adds a second OS boundary, a second credential store, and path-translation hazards (`/mnt/c/...` vs `C:\...`) for zero benefit on a Windows-native fleet. Increases the 2am surface area rather than shrinking it.

## Decision Criteria
1. **2am legibility** — can on-call see *what ran, when, with what exit code* without reverse-engineering the scheduler config?
2. **Native environment fidelity** — runs where the repos actually live, with the real credential and path context (incl. OneDrive - Blue Yonder exclusions).
3. **Version control of behavior** — the *logic* must be diffable and reviewable, not buried in a Task Scheduler XML blob.
4. **Idempotency & re-runnability** — a human must be able to run the exact same thing by hand.
5. **Failure visibility** — non-zero exits surface somewhere a person will actually look.

## Assumptions
- The run genuinely needs local machine state (repo working copies, local creds, machine-specific paths); if it's actually portable and stateless, CI wins instead.
- This is a single-host operation, not fan-out across many machines (no fleet-orchestration tool like Ansible in play).
- The machine is up / can wake for the schedule, or missed-run catch-up is acceptable.

## Recommendation
**Now:**
- Write **`fleet-baseline.ps1`** in a version-controlled ops repo. It takes zero required args, discovers repos from a checked-in manifest (explicit list, not glob), and is safe to run twice.
- Register **one Task Scheduler task** whose action is `pwsh -NoProfile -File <path>\fleet-baseline.ps1`. Set "Run whether user is logged on or not," enable **"Run task as soon as possible after a scheduled start is missed."**
- Have the script write a **transcript log per run** to a fixed local path (`Start-Transcript`), and emit a clear final line: `BASELINE OK` / `BASELINE FAIL <repo>`.
- Set the task to **`Send a notification / non-zero exit → alert`** via the script itself (email/Teams webhook on failure), not via scheduler's own weak history.

**Not now:**
- Don't put per-repo logic, secrets, or paths into scheduler arguments.
- Don't split into N scheduled tasks (one per repo) — that fragments failure visibility.
- Don't route reads through any `OneDrive - Blue Yonder` path.

**Defer:**
- A CI scheduled workflow that runs the same script's *check-only* mode as a drift detector, once the local run is stable.
- Multi-host orchestration — only if the fleet grows beyond one machine.

## Rationale
- **The scheduler is a dumb trigger; the script is the contract.** Anyone reading `fleet-baseline.ps1` in git sees the entire behavior. Task Scheduler XML is where logic goes to die at 2am.
- **Local run avoids reconstructing state you already have.** CI would need to re-clone every repo and re-provision creds each run — added minutes and a whole new failure class, for a run that by definition wants the local fleet.
- **One task = one place to look.** Exit code + transcript + explicit final-line status collapses triage to a single log tail.
- **Idempotency lets on-call reproduce the failure by hand** — the single most valuable property when the scheduled run breaks and you can't see why.
- **Missed-start catch-up + wake handling** covers the dominant real-world failure of local schedulers: the machine was asleep or off.

## Trade-offs
- You give up CI's clean audit trail and centralized history — mitigated by the per-run transcript, but it's local, so log rotation/retention is now *your* job.
- Single-host coupling: this run is tied to one machine's uptime. That's the cost of "local."
- Task Scheduler's own reliability is mediocre; you're leaning on the script's logging rather than the scheduler's reporting.

## Risks
- **Silent failure (machine asleep / task disabled):** likelihood medium, impact high. Mitigate with a **heartbeat** — the script writes a `last-success` timestamp; a lightweight check (or the deferred CI job) alerts if it's stale > interval.
- **Credential/context drift when "run whether logged on or not":** likelihood medium, impact medium. Test explicitly under the service context, not just interactively — a script that works logged-on can fail headless.
- **Log growth / disk fill:** likelihood low, impact medium. Add transcript rotation (keep N runs).
- **Manifest rot** (new repo added, never baselined): likelihood high over time, impact medium. Make the script **fail loudly on unknown/missing repos** rather than skipping silently.

## Signals to Revisit
- The run becomes **genuinely stateless/portable** → move to CI scheduled workflow as primary.
- Fleet grows to **multiple hosts** → promote to a real orchestrator (Ansible/scheduled runner pool).
- Task Scheduler proves **unreliable for wake/catch-up** in practice → move the trigger to a lightweight always-on service or a CI cron that SSHes/remotes in.
- On-call reports they **can't reproduce failures by hand** → the idempotency assumption broke; fix the script before adding tooling.


### Openai (gpt-5.6-terra)

## Position

Use a dedicated Windows Task Scheduler task running a versioned PowerShell entrypoint. It is the lowest-operations mechanism that reliably supports recurring local, cross-repo execution.

## Alternatives Considered

- A long-running Windows service: rejected because it adds deployment, lifecycle, logging, and recovery burden disproportionate to a scheduled baseline job.
- GitHub Actions or another CI scheduler: rejected because the run is explicitly local and likely needs access to local repositories, credentials, and machine-specific state.

## Decision Criteria

- Must run reliably on the local Windows host without manual intervention.
- Must access multiple local repositories and any required local tooling.
- Must be operable by a small team without service-management expertise.
- Must provide clear logs, failure visibility, and safe reruns.
- Must be quick to ship and easy to change as repositories evolve.

## Assumptions

- The baseline can run unattended in a bounded window and does not require an interactive desktop.
- A stable local account can access every target repository and required credentials.
- The host is normally powered on at the scheduled time, or catch-up execution is acceptable.
- The job’s runtime is under the chosen recurrence interval.

## Recommendation

- What to do now: create one Task Scheduler task, owned by a dedicated local service account where practical, that invokes a checked-in PowerShell runner from a small control repository. Configure “run whether user is logged on or not,” retry on failure, prevent overlapping runs, and write timestamped logs plus an exit-code summary.
- What not to do now: do not create a custom daemon, Windows service, or distributed orchestrator.
- What to defer until later: central scheduling, fleet-wide aggregation, dashboards, and CI-based execution once multiple hosts or stronger audit requirements justify them.

## Rationale

- Task Scheduler is already available on the target platform; implementation is typically hours, not days.
- A single runner gives one place to define repository order, timeouts, environment checks, and failure policy.
- Cross-repo local access remains straightforward; no repository synchronization or remote credential redesign is required.
- “Do not start a new instance” prevents concurrent runs from corrupting worktrees or competing for shared tools.
- Structured local logs and non-zero exits provide enough operational visibility for an initial fleet-baseline process.

## Trade-offs

- You give up centralized observability and uniform execution across machines.
- Failure notification requires a small additional integration or periodic log review.
- Task definitions can drift from source control unless exported or provisioned through a script.

## Risks

- **Credentials or mapped drives unavailable in non-interactive execution** — likelihood: medium; impact: high. Mitigate by using absolute local paths, a dedicated account, and preflight checks.
- **A run overlaps or exceeds its schedule window** — likelihood: medium; impact: medium. Mitigate with single-instance enforcement, per-repo timeouts, and a runtime alert threshold.
- **Silent failures go unnoticed** — likelihood: medium; impact: medium. Mitigate with retained logs, explicit summary output, and a notification on non-zero exit.
- **Scheduler configuration drifts between hosts** — likelihood: low initially; impact: medium. Mitigate by creating the task through a versioned provisioning script.

## Signals to Revisit

- The run must execute on more than 3–5 hosts.
- Central audit trails, RBAC, or compliance evidence becomes mandatory.
- Failure response needs paging or guaranteed notification.
- The job routinely runs longer than its recurrence interval.
- Repository access or dependencies move from local machine state to centrally managed infrastructure.


## Round 2: Critique

### Claude (claude-opus-4-8)

# Evaluation

Both proposals converge on the same skeleton — **Task Scheduler as trigger, versioned PowerShell as the contract.** That convergence is itself a signal: it's the obvious right answer for a single-host, local, cross-repo run. The interesting differences are in what each *ignores*, and at 2am the ignored things are what page you.

---

## Proposal A

**Steelman.** If the run genuinely depends on local machine state — real working copies, machine-local creds, OneDrive-excluded paths — and the fleet is exactly one host, then A is close to optimal. Its core insight is correct and load-bearing: *the scheduler is a dumb trigger; the script is the contract.* The heartbeat / `last-success` timestamp is the single best idea in either proposal, because it's the only mechanism proposed that survives the dominant real-world failure: nothing ran at all.

**Assessment: Partially agree.** The Position is right. But A quietly promotes itself from "pick a host" to "here is the whole implementation," and in doing so smuggles in choices that deserve scrutiny — email/Teams webhook on failure, transcript rotation, manifest discovery. That's fine as a sketch; the danger is treating a 40-line answer as a settled design.

**Strongest point.** The heartbeat/staleness check. A scheduled job's worst failure is *absence*, and absence is invisible to every mechanism that only reports on runs that happened. A is the only proposal that closes this loop.

**Weakest assumption.** "The run genuinely needs local machine state." A leans hard on this to reject CI, but never tests it. If the "cross-repo baseline" is actually reading committed files and producing a report — portable, stateless — then the entire local-hosting argument collapses and CI wins on audit trail alone. A hedges this ("if it's actually portable, CI wins instead") but doesn't do the 10-minute analysis to find out. That's the whole ballgame, left as an aside.

**Hidden assumptions.**
1. **That failure-on-non-zero-exit implies failure-is-noticed.** A wires alerting *inside the script itself* — email/Teams on failure. But a script that dies (unhandled exception, killed process, hung on a network path) never reaches its own alert line. The alert path shares fate with the thing it's monitoring. This is a classic monitoring inversion, and the heartbeat is the *only* thing that partially rescues it — which means the heartbeat isn't a nice-to-have, it's the actual safety net, and A files it under "Not now / Defer to CI."
2. **That `Start-Transcript` captures what broke.** Transcript catches stdout/stderr of the PowerShell session. Native tool failures (git spawning a pager, a credential prompt on a headless session, a child process writing to a raw handle) often escape it. A treats "we have a transcript" as "we have the diagnostic," which is optimistic.

**Overlooked risks.** *Partial completion.* A's status line is binary — `BASELINE OK` / `BASELINE FAIL <repo>`. What about repo 4 of 9 succeeding, repo 5 timing out, repos 6–9 never attempted? Idempotency helps rerun, but the *state after a partial run* is undefined and unaddressed. Also: **concurrency.** A never mentions preventing overlapping runs — if a run hangs past its interval, the next fires on top of it, and two processes contend for the same worktrees. B caught this; A didn't.

---

## Proposal B

**Steelman.** B is right where it's most conservative: it names *single-instance enforcement* ("do not start a new instance"), *preflight checks*, and *provisioning the task through a versioned script*. If the operating team is small and non-specialist, B's "lowest-operations mechanism" framing is exactly the correct decision axis — it optimizes for the thing that actually determines survival at 3 years: how much standing expertise the mechanism demands.

**Assessment: Partially agree.** Same correct Position, but B is vaguer than A on the one thing that matters most — how anyone *knows* it failed. "A small additional integration or periodic log review" is not a monitoring strategy; "periodic log review" is how silent failures become quarterly surprises.

**Strongest point.** "Do not start a new instance" + task-created-through-a-provisioning-script. These two together address the two most common operational rots: worktree corruption from overlap, and config drift from hand-edited scheduler XML. B is more disciplined than A here.

**Weakest assumption.** "The job's runtime is under the chosen recurrence interval." B lists this as an assumption *and* as a risk, but never resolves the tension. Cross-repo runtime is unbounded in practice — one repo grows, one network path stalls, and the run that took 8 minutes now takes 90. B's single-instance guard means the *next* run silently no-ops. Combine that with "periodic log review" and you get a job that has effectively stopped running while every individual run that *did* run looks green.

**Hidden assumptions.**
1. **That a "dedicated local service account" is available and low-cost.** B mentions it "where practical" — in a corporate Windows environment (this is Blue Yonder tooling), service accounts often require IT tickets, password-rotation policy, and Group Policy interaction. B treats the account as a config checkbox; it can be the longest-lead item in the whole project.
2. **That "operable by a small team without service-management expertise" and "run whether logged on or not" coexist freely.** They're in tension. Headless/service-context execution is *precisely* the mode that requires the most operational sophistication to debug (no desktop, no interactive cred prompt, different profile). B picks the low-expertise team *and* the high-expertise execution mode without noticing the conflict.

**Overlooked risks.** No heartbeat. B's entire failure-visibility story is reactive — logs exist, non-zero exit emits something, humans review periodically. Nothing detects *the run that never happened*. This is B's single biggest gap versus A, and it's the failure mode with the highest 2am cost.

---

# Revised Recommendation

**I maintain my position, and the two proposals sharpened rather than changed it.** Both correctly land on Task Scheduler + versioned PowerShell; neither gave me reason to move off it, because for a single local Windows host it is the mechanism with the lowest standing-expertise requirement — the metric that governs 3-year operability.

What changed: **I now treat three things A and B list as optional/deferred as mandatory, non-negotiable parts of the answer.**

1. **The heartbeat is not "Defer." It is the primary failure detector.** (A's best idea, B's biggest gap.) The script writes a `last-success` timestamp on clean completion. A *separate, independent* checker — not the same script — alerts when it goes stale past interval. This is the only mechanism that survives the dominant failure: nothing ran. In-script alerting (A) and periodic log review (B) both share fate with the thing they monitor and both miss silent absence. Reject both as the primary signal.

2. **Single-instance enforcement is mandatory, not a nicety.** (B caught it, A missed it.) "Do not start a new instance," plus a per-repo timeout so one stalled repo can't consume the whole window. Without this, an overrun turns into either worktree contention (A) or silent no-op runs (B).

3. **The task must be created by a checked-in provisioning script, and status must be tri-state, not binary.** `OK` / `FAIL <repo>` / **`PARTIAL <last-completed-repo>`**. A's binary status hides the partial-completion case, which is the most common real outcome of a cross-repo loop.

**The one thing I'd resolve before writing any of it** — and neither proposal did — **is A's weakest assumption: does this run actually need local state?** If the baseline reads committed content and emits a report, it's portable, and CI's free audit trail beats everything above. That's a 10-minute determination that flips the entire host choice. Do it first. Both proposals asserted "it's local" as a premise rather than a finding; I won't.

**Net:** Host = Task Scheduler + versioned PowerShell (unchanged). But the deliverable is *four* checked-in artifacts, not one: the runner, the provisioning script, an independent heartbeat checker, and the manifest. A treated three of those as deferrable; that's the design's actual attack surface at 2am, so they ship in v1 or the mechanism isn't done.


### Openai (gpt-5.6-terra)

### Proposal A

**Steelman first:** Exactly right if the job truly depends on one Windows host’s local repositories, tools, and credentials; the team can maintain one small PowerShell control plane; and a local alert/heartbeat mechanism is acceptable operationally.

**Your assessment:** **Agree.** This is the right mechanism for a single-host, local cross-repo job. Task Scheduler provides the trigger with essentially zero platform work; the version-controlled script keeps the behavior reviewable and runnable by humans.

**Strongest point:** “The scheduler is a dumb trigger; the script is the contract.” That sharply separates configuration from operational logic.

**Weakest assumption:** That “run whether logged on or not” has equivalent access to the required repositories, credentials, certificates, network locations, and tooling. If false, the task succeeds interactively but fails unattended—the exact mode that matters.

**Hidden assumptions:**

- The task registration itself is provisioned/versioned, rather than hand-maintained Scheduler UI state.
- Running a missed job later is always safe; that may be false if the baseline changes working trees or consumes constrained resources.

**Overlooked risks:**

- A baseline that modifies repositories can collide with developers’ work, IDEs, Git operations, or repository locks.
- Executing commands across local repositories is a supply-chain boundary: an altered repository script can gain execution under the scheduled account.
- Alert delivery can fail independently; a local transcript plus a webhook is not guaranteed notification.

### Proposal B

**Steelman first:** Exactly right if this is a bounded, non-interactive job on one stable host, with a dedicated account that has all necessary local access, and the team needs an implementation in hours—not a new operational platform.

**Your assessment:** **Partially agree.** Its core choice is correct, but “dedicated service account where practical” understates the main operational risk. For local working copies, a service account may see a different profile, credential store, SSH configuration, package environment, and permissions than the person who owns the repos.

**Strongest point:** Explicitly requiring retry behavior and single-instance enforcement. Preventing overlap is essential for cross-repository worktree safety.

**Weakest assumption:** That a stable unattended account can access every dependency. If it cannot, the task becomes unreliable or teams weaken security by sharing user credentials.

**Hidden assumptions:**

- All repository actions are safe to run without an interactive session or approval prompt.
- The job completes within its interval even after retries, slow network calls, package updates, or a partially failed prior run.

**Overlooked risks:**

- No explicit stale-success/heartbeat detection; retries do not reveal a disabled task, broken trigger, or persistent failure.
- No stated policy for dirty worktrees, uncommitted changes, or repositories that have been moved or removed.
- Task drift is noted but not made a launch requirement; manual Scheduler configuration will drift.

## Revised recommendation

Use **one Windows Task Scheduler task invoking a version-controlled PowerShell orchestrator**, provisioned by a version-controlled registration script. This remains the best answer: both proposals correctly identify that a local, single-host cross-repo run should not be made into a CI or service project.

Ship only after these operational guardrails are in place:

- Run as the **least-privileged account that actually owns or can safely access the local repositories**; validate the exact unattended execution context before enabling recurrence.
- Use an explicit repository manifest, preflight checks, per-repo timeouts, non-overlap enforcement, and a defined dirty-worktree policy—prefer fail-closed unless mutation is explicitly intended.
- Log structured per-run results locally, retain/rotate them, return non-zero on failure, and maintain a `last-success` heartbeat monitored independently of the task.
- Provision the task through code; do not rely on a manually edited Scheduler task.
- Treat “run as soon as possible after a missed start” as opt-in: enable it only after confirming a delayed execution is safe.

I would not adopt Proposal A’s automatic catch-up as a default. The compelling local-script architecture does not justify running potentially stateful cross-repo work at an arbitrary later time. Enable catch-up only when the baseline is demonstrably read-only or idempotent against active developer worktrees.


## Synthesis (by gemini, non-participant)

## Consensus
Both participants reached a strong, independently reasoned consensus on the core architecture: **Windows Task Scheduler triggering a version-controlled PowerShell script**. 

This consensus was driven by tight shared reasoning regarding the specific constraints of the problem:
- CI/CD (like GitHub Actions) is the wrong tool for operations requiring local machine state (worktrees, local credentials, excluded paths), as it forces you to reconstruct state you already possess.
- Windows Services or distributed orchestrators are too heavy for a single-host, cross-repo task and require disproportionate operational expertise.
- **The scheduler must be a "dumb trigger" and the script the "contract."** All logic, manifests, and configurations must live in version control, not in the Task Scheduler UI XML.
- **Single-instance enforcement is mandatory.** Overlapping runs will corrupt local worktrees.
- **Task provisioning must be scripted.** The scheduled task itself must be created via a version-controlled script to prevent configuration drift.

## Unresolved Disagreements

**1. Handling Missed Executions (Catch-up Runs)**
*   **The Crux:** Should the task be configured to "Run as soon as possible after a scheduled start is missed"? Claude argued this is mandatory because the dominant failure mode of local schedulers is the machine being asleep. OpenAI argued this is highly dangerous because a delayed, unpredictable run could execute while a developer is actively working.
*   **Stronger Argument:** OpenAI. Automatically executing stateful, cross-repo operations the moment a user wakes their machine risks colliding with active IDEs, repository locks, and uncommitted work. Catch-up execution should strictly be opt-in and only enabled if the baseline is proven mathematically read-only.

**2. Execution Context and Credentials**
*   **The Crux:** OpenAI advocated for a "dedicated local service account where practical" to ensure stable unattended access. Claude pointed out that service accounts in corporate environments require heavy IT overhead, and running "whether logged on or not" introduces severe headless-execution credential hurdles (no interactive prompts, different profile).
*   **Stronger Argument:** Claude. For a local fleet-baseline operating on developer working copies, introducing a separate service account divorces the script from the exact local state (mapped drives, SSH agents, user-specific Git configs) it was designed to measure. 

## Argument Quality Assessment
The overall reasoning in both proposals was exceptionally high, with both participants accurately identifying the flaws in their own initial proposals during Round 2. 

*   **Single strongest argument:** Claude's observation regarding failure visibility: *"Absence is invisible to every mechanism that only reports on runs that happened."* This perfectly articulated why in-script alerts (webhooks) and periodic log reviews are insufficient, proving that an external heartbeat monitor is a non-negotiable architectural requirement.
*   **Single weakest argument:** OpenAI's initial suggestion in Round 1 to rely on "periodic log review" to catch failures. This is a well-known anti-pattern that guarantees silent failures will eventually be ignored. (OpenAI correctly abandoned this in Round 2).

## Blind Spots
*   **Is local execution actually required?** Claude flagged this but neither participant resolved it. If the baseline merely reads committed files to emit a report, it is stateless. If it is stateless, the entire premise of the debate is moot, and a CI scheduled workflow is vastly superior due to its native audit trails and isolated environments.
*   **Dirty Worktree Policy:** What happens if the script runs while uncommitted changes exist? Neither proposal defined a strict policy (e.g., fail-closed vs. stash-and-restore) for handling dirty local state.
*   **Supply Chain / Privilege Escalation:** Running an automated script across multiple local repositories assumes all local repositories are safe. If the baseline executes any local repository code (e.g., `npm run build`), a compromised repo can execute arbitrary code under the scheduler's privileges.

## Recommended Decision
**Host the run using Windows Task Scheduler to invoke a version-controlled PowerShell script, conditionally.**

**Prerequisite:** Perform a 10-minute audit to confirm the job genuinely requires uncommitted local state, machine-specific paths, or local credentials. If it does not, abort this local design and put the job in CI. 

Assuming local execution is required, implement the solution as a package of four artifacts, not just one script:
1.  **The Runner:** A PowerShell script (`fleet-baseline.ps1`) that executes the logic, outputs a tri-state status (`OK`, `FAIL <repo>`, `PARTIAL <repo>`), enforces per-repo timeouts, and writes a `last-success` timestamp file upon clean completion.
2.  **The Manifest:** A checked-in, explicit list of target repositories (no dynamic globbing).
3.  **The Provisioner:** An Infrastructure-as-Code script that registers the scheduled task, enforcing "Do not start a new instance" (single-instance lock).
4.  **The Heartbeat Monitor:** A completely independent lightweight check that alerts if the `last-success` timestamp goes stale.

**Key rejections of proposed ideas:**
*   Do *not* enable automatic missed-start catch-up.
*   Do *not* use a separate service account unless absolutely necessary; run under the context of the user whose worktrees are being baselined, but explicitly test it in headless mode.

## Risks
*   **Headless Context Failures:** Scripts that work in an interactive terminal often fail in Task Scheduler due to missing SSH keys, different environment variables, or invisible authentication prompts.
    *   *Mitigation:* Pre-flight checks at the top of the script asserting required paths and credentials exist before taking action.
*   **Silent Task Death:** The scheduler gets disabled or the script hangs indefinitely, preventing future runs.
    *   *Mitigation:* The external heartbeat monitor. If the timestamp doesn't update, page the team.
*   **Developer Interference:** The script mutates a repository while a developer is working on it.
    *   *Mitigation:* Implement a "dirty worktree" check. The script must loudly fail or skip any repository with uncommitted changes or active locks.

## Action Items
1.  **Validate Statefulness:** Confirm the script cannot be run in CI.
2.  **Draft the Script & Manifest:** Write the idempotent PowerShell runner and explicitly list the target repositories. Include a fail-closed check for dirty Git worktrees.
3.  **Create the Provisioner:** Write a script to create the Task Scheduler XML definition. Hardcode single-instance enforcement and disable catch-up runs.
4.  **Test Headless Execution:** Manually trigger the scheduled task while logged off to verify credential and path resolution.
5.  **Deploy Heartbeat Alert:** Set up an external monitor (e.g., Datadog, a separate lightweight CI cron, or simple monitoring daemon) to alert if the `last-success` file is older than the recurrence interval + buffer.
