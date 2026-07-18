# AI Council Debate: What mechanism should host the recurring local cross-repo fleet-baseline run?

**Panel:** claude-opus-4-8, gpt-5.6-terra
**Synthesizer:** gemini (non-participant)
**Rounds:** 2
**Panel Mode:** custom
**Debate Mode:** pick
**Source:** cli

## Verdict Summary

**Decision:** Host the recurring run via a scheduled CI pipeline in a dedicated `fleet-baseline` repository, using ephemeral hosted runners by default.
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

Host the recurring cross-repo fleet-baseline run as a scheduled CI pipeline job (e.g., GitHub Actions scheduled workflow or GitLab scheduled pipeline) in a dedicated "fleet-baseline" repository, not on individual developer machines or an ad-hoc cron on a shared box. If the run needs privileged fleet access, back the scheduled job with a self-hosted runner rather than moving the orchestration off CI.

## Alternatives Considered

- **Cron on a shared VM/bastion host.** Rejected. Cron jobs are invisible until they silently stop. There is no built-in run history, no log retention, no ownership trail, and no alerting. The 2am on-call engineer inherits a script whose last successful run and last edit are both unknowable without SSH archaeology. Cognitive load is high and the failure mode is silent drift.
- **A dedicated orchestrator (Airflow / Temporal / Nomad periodic job).** Rejected *for now*. These are the right answer once you have a dozen interdependent scheduled workflows with DAG dependencies, backfills, and SLAs. For a single recurring baseline run, they add an entire stateful service to operate, patch, and back up — a large fixed operability cost to solve a problem you don't yet have.

## Decision Criteria

1. **2am debuggability.** Can on-call see the last run, its logs, and its exit status in under 60 seconds without SSH?
2. **Ownership clarity.** Is it obvious which team/repo owns the job and how to change its schedule via reviewed code?
3. **Fixed operational overhead.** No new stateful service to patch/back up/monitor unless justified.
4. **Auditability & reproducibility.** Runs are versioned, logged, and retriable with a button.
5. **Secrets/access handling.** Credentials for cross-repo/fleet access are managed, scoped, and rotatable — not baked into a host.

## Assumptions

- The run is genuinely recurring and roughly single-task (one baseline sweep), not a growing web of dependent scheduled jobs. If it's already a DAG, the orchestrator answer wins.
- Your org already operates a CI platform (Actions/GitLab/Buildkite) with scheduled-trigger support and secret management. If not, that changes the build-vs-adopt math.
- The run fits within CI job timeout limits (typically 6h on hosted runners) or you control a self-hosted runner. A multi-day run needs different hosting.
- Cross-repo access can be granted via a scoped machine identity (GitHub App / deploy token / service account), not a human's credentials.

## Recommendation

**Do now:**
- Create a `fleet-baseline` repo holding the run script *and* the schedule definition as code (`.github/workflows/baseline.yml` or equivalent).
- Trigger via native scheduled workflow + a `workflow_dispatch`/manual trigger so on-call can re-run with one click.
- Grant fleet/cross-repo access via a scoped machine identity stored in the CI secret store; document the scope in the repo README.
- Configure failure alerting to the owning team's on-call channel (not email).
- Set explicit log retention (≥30 days) and pin the runner image.

**Don't do now:**
- Don't stand up Airflow/Temporal for one job.
- Don't run it from anyone's laptop or a hand-maintained cron.
- Don't grant the job broad, long-lived admin credentials — scope to what the baseline actually reads/writes.

**Defer:**
- Migration to a dedicated orchestrator until you have 3+ interdependent scheduled jobs, backfill needs, or per-run SLAs that CI can't express.
- Custom dashboards until failure rate/duration variance justifies them.

## Rationale

- **Failure visibility is the whole ballgame.** CI gives you run history, per-step logs, and a red/green status in the UI for free. Cron gives you nothing until you build it — and you won't build it until after the first silent 3-week outage.
- **Schedule-as-code closes the "who changed this" gap.** A cron edit is a `crontab -e` with no reviewer and no diff. A workflow schedule change is a PR. In a 3-year horizon this is the difference between traceable and archaeological.
- **Manual re-trigger is the on-call's escape hatch.** `workflow_dispatch` turns a 2am incident from "SSH to a host, find the script, figure out the args, run it" into "click Re-run." That's minutes vs. tens of minutes under stress.
- **Self-hosted runner preserves the CI benefits while solving network/privilege needs.** You get fleet-network access without discarding history, logging, and alerting — the common excuse for going back to cron.
- **Zero new stateful infra.** No orchestrator DB to back up or upgrade. The marginal operability cost of one more scheduled workflow is near zero.

## Trade-offs

- **Scheduler precision & DAG expressiveness.** CI cron is best-effort and can be delayed by minutes under queue load; it can't express "run B only after A succeeds across repos" cleanly. You give this up vs. an orchestrator — acceptable for a single baseline run.
- **Job duration ceiling.** Bound by CI timeouts (~6h hosted). Long sweeps need a self-hosted runner or chunking.
- **Vendor coupling.** The schedule lives in your CI platform's syntax; a future migration is real work. Keep the actual logic in a portable script (not YAML) to cap this cost.

## Risks

- **Silent schedule disablement.** GitHub disables scheduled workflows after 60 days of repo inactivity. *Likelihood: high over 3 years. Impact: high (silent stop).* Mitigate with a liveness check: alert if no successful run in N+1 intervals, monitored from *outside* the workflow.
- **Secret sprawl / over-broad access.** A fleet-wide credential in CI is a juicy target. *Likelihood: medium. Impact: high.* Mitigate with scoped machine identity, short-lived tokens (OIDC where possible), and rotation.
- **Runner starvation / queue delay causing missed windows.** *Likelihood: medium. Impact: low-medium.* Mitigate with a dedicated runner label and duration-drift alerting.
- **Growth into a DAG without re-evaluating.** Teams bolt "just one more step" onto the baseline job. *Likelihood: high. Impact: medium.* Mitigate with the explicit revisit signal below.

## Signals to Revisit

- You accumulate **3+ interdependent scheduled jobs** or need cross-job dependencies/backfills → move to a real orchestrator.
- Run duration exceeds CI timeout or requires **>1 day** of execution → dedicated compute + orchestrator.
- **Missed-run rate** climbs (schedule reliability degrades) despite liveness monitoring.
- Per-run **SLA/latency guarantees** become a business requirement CI cron can't meet.
- Secret-access scope must broaden to true fleet-admin → warrants a hardened, purpose-built execution environment with tighter controls than shared CI.


### Openai (gpt-5.6-terra)

## Position
Host the recurring local cross-repo fleet-baseline run on a dedicated always-on utility runner using a `systemd` service and timer. Keep orchestration in a versioned control repository and make the host execution environment intentionally simple and observable.

## Alternatives Considered
1. **Scheduled CI workflow (for example, GitHub Actions/GitLab CI scheduled pipeline):** Rejected because a local cross-repo run commonly needs a durable workspace, local tooling/cache, and controlled access to multiple repositories. CI is viable later, but debugging scheduled runner failures and managing cross-repo credentials adds unnecessary coupling now.

2. **Kubernetes `CronJob`:** Rejected because it introduces cluster operations, image lifecycle, secret management, persistent workspace behavior, and debugging overhead disproportionate to a recurring baseline task. It is only justified if the job needs horizontal scale, isolation, or existing Kubernetes operations ownership.

## Decision Criteria
- The current team must be able to build, operate, and debug the mechanism without dedicated platform engineering support.
- The run must access and coordinate multiple repositories reliably from a local/shared workspace.
- Missed runs, overlapping runs, failures, and credential expiration must be visible and recoverable.
- The mechanism should have low operational surface area and minimal new infrastructure.
- The job should be reproducible enough to migrate to CI or containerized execution later.

## Assumptions
- The run is periodic (for example, daily or weekly), not latency-sensitive or event-driven.
- It can execute on one controlled, always-on machine or self-hosted runner with network access to all relevant repositories.
- The job’s output can be expressed as logs, a summary artifact, and preferably pull requests rather than unreviewed direct pushes.
- Expected execution time is under roughly 1–2 hours and does not require parallel execution across many hosts.

## Recommendation
**What to do now**
- Provision or designate one managed utility VM/self-hosted runner for this job; do not run it from an engineer’s workstation.
- Implement the run as a versioned script or small CLI in a control repository, with pinned tool versions and explicit repository manifests.
- Invoke it through a `systemd` oneshot service and `systemd` timer:
  - Use `Persistent=true` so a missed run executes after host recovery.
  - Use a randomized delay to avoid predictable contention with backups or other scheduled work.
  - Use an exclusive lock (`flock` or equivalent) to prevent overlapping runs.
  - Set explicit runtime limits and failure behavior.
- Use a dedicated machine/service identity with least-privilege repository access. Rotate credentials through the existing secret-management mechanism.
- Send a concise success/failure summary to the team’s existing operational channel; alert on failure and on no successful run within the expected interval.
- Have the job create reviewable PRs or reports by default. Require an explicit, separately governed path for direct fleet-wide changes.

**What not to do now**
- Do not introduce Kubernetes, a distributed scheduler, or a bespoke orchestration service.
- Do not rely on user crontabs, laptops, shared personal credentials, or manually maintained local checkouts.
- Do not allow concurrent runs or automatic direct pushes across repositories without review and rollback controls.

**What to defer until later**
- Migration to scheduled CI or a containerized worker once the workflow is stable and its dependency/tooling requirements are understood.
- Parallelization, sharding, and queue-based orchestration until run duration or repository count makes a single runner inadequate.
- Fully automated remediation beyond PR generation until baseline changes have demonstrated low false-positive and rollback rates.

## Rationale
- A `systemd` timer provides operationally meaningful behavior that plain cron lacks: persisted missed-run handling, service status, journal logs, resource limits, and standard restart diagnostics.
- One utility runner keeps the cross-repo workspace and toolchain stable. For a job touching tens of repositories, this is materially easier to diagnose than ephemeral CI environments with separate checkout and credential behavior.
- The implementation is small: typically one VM, one service unit, one timer unit, one repository manifest, and one alert integration—generally days of work rather than weeks of platform work.
- Exclusive execution matters more than horizontal scale initially: two overlapping baseline runs can generate duplicate PRs, race on shared worktrees, or overwrite generated state.
- Producing PRs rather than direct writes makes the automation useful immediately while limiting blast radius across the fleet.

## Trade-offs
- The utility runner is a single execution point; host outages can delay a run until recovery. `Persistent=true` reduces missed-run risk but does not provide high availability.
- This is less elastic than CI or Kubernetes. A 90-minute run remains a 90-minute serial run unless explicitly redesigned.
- The team accepts responsibility for patching one managed host and monitoring disk, credentials, and runner health.
- Reproducibility is weaker than a fully containerized execution environment, although pinned tool versions and a bootstrap script substantially reduce drift.

## Risks
- **Host failure or disk exhaustion — Likelihood: medium; Impact: medium.** Mitigate with managed-host patching, disk monitoring, a documented rebuild procedure, and a clean-workspace/bootstrap command.
- **Credential expiration or excessive permissions — Likelihood: medium; Impact: high.** Use a dedicated service identity, least-privilege repository scopes, expiry monitoring, and documented rotation ownership.
- **A bad baseline update creates broad unwanted changes — Likelihood: medium; Impact: high.** Default to report/PR generation, cap the number of repositories changed per run, require CI validation on generated PRs, and retain a rollback procedure.
- **Run duration grows until schedules overlap — Likelihood: medium as repository count grows; Impact: medium.** Record duration per run, enforce a lock and timeout, and redesign for sharding only when median runtime exceeds roughly 50% of the scheduled interval.
- **Silent failures due to timer or notification misconfiguration — Likelihood: low to medium; Impact: high.** Alert both on explicit failure and on absence of a successful completion within 1.5–2 expected intervals.

## Signals to Revisit
- Median runtime exceeds 50% of the recurrence interval, or the job regularly exceeds its timeout.
- The job needs more than one runner for throughput, isolation, or availability.
- Multiple teams require independent schedules, permissions, or execution environments.
- The job becomes event-driven rather than periodic, such as needing to react immediately to repository changes.
- The organization already operates scheduled CI runners or Kubernetes workloads reliably enough that migrating removes operational burden rather than adding it.
- The run requires stronger reproducibility, supply-chain controls, or auditability than a managed utility host can provide.


## Round 2: Critique

### Claude (claude-opus-4-8)

# Evaluation

## Proposal A: systemd service + timer on a dedicated utility runner

**Steelman first:**
This is exactly right if the org does *not* already operate a reliable CI platform with scheduled triggers, or if that CI platform cannot grant the network reachability and credential scope this job needs without contortions. It's also right if the run genuinely depends on a durable, warm workspace — large multi-repo checkouts, populated caches, local toolchains that are expensive to reconstruct — where ephemeral execution would dominate runtime. If the team owning this already runs Linux hosts competently and treats one more service unit as free, then systemd's `Persistent=true`, `flock`, resource limits, and journald give you real operational primitives that plain cron lacks. In that world, A is the low-drama answer.

**Your assessment:** Partially agree.
The instinct to reject Kubernetes and distributed schedulers is correct, and the operational hygiene (locking, missed-run handling, least-privilege identity, PRs over direct pushes) is genuinely good. But the Position under-weights the single thing my 2am lens cares most about: *default observability without SSH*. A's failure story is "check `systemctl status`, read the journal" — which means SSH to a specific host, on a specific machine, with the right access, at 2am. That is precisely the archaeology Proposal B rightly attacks.

**Strongest point:**
The exclusive-execution argument. "Two overlapping baseline runs can generate duplicate PRs, race on shared worktrees, or overwrite generated state" is the correct thing to be paranoid about, and `flock` is the honest, boring answer. B never seriously addresses concurrency control — it assumes CI handles it, which is not always true across self-hosted runners.

**Weakest assumption:**
"The team accepts responsibility for patching one managed host and monitoring disk, credentials, and runner health." This is stated as a trade-off but treated as small. In practice, a single-purpose utility VM is exactly the box that rots: nobody remembers it exists, the OS falls behind on patches, the disk fills with worktrees, and the person who built it leaves. In three years this host *is* the technical debt. If that assumption is false — and it usually is — you get silent drift plus a security liability.

**Hidden assumptions:**
1. That the run history and logs living *on the host* (journald) is acceptable. It never states that observability is host-local, but it is — and that couples debuggability to host reachability and host survival. If the host dies, so does your run history.
2. That "one utility runner" stays singular. The proposal assumes a stable ownership model, but a general-purpose utility VM attracts other cron-shaped jobs. Nothing in the design prevents it from becoming the shared-box-of-mystery that everyone (including A) says to avoid.

**Overlooked risks:**
- **Ownership decay of the host itself.** A's risks cover disk, credentials, and bad baselines, but not "who owns this VM after the author leaves." There's no reviewed-code trail for the *host configuration* — only for the script in the control repo. The unit files, the flock setup, the timer tuning live on the box unless explicitly captured in config management, which A doesn't mandate.
- **No manual re-trigger story.** A does not give on-call a one-button re-run. Recovery means SSH + invoke-the-service-manually + hope the args match what the timer used.

---

## Proposal B: scheduled CI pipeline in a dedicated repo

**Steelman first:**
This is exactly right if the org already runs a CI platform with scheduled triggers and secret management (B states this as an assumption, correctly), and if the run fits within CI's execution model — either under the timeout on hosted runners or on a self-hosted runner for privilege/network needs. In that world you get run history, per-step logs, red/green status, schedule-as-code, and a `workflow_dispatch` escape hatch *for free*, with zero new stateful infrastructure. For the 2am engineer, "open the Actions tab, see the last run" beats "SSH and grep journald" decisively.

**Your assessment:** Agree, with one hard caveat.
B correctly identifies that failure visibility is the whole game and that schedule-as-code closes the "who changed this" gap — both are exactly my priorities. The caveat: B waves at concurrency and durable-workspace concerns that A takes seriously, and the self-hosted-runner path quietly reintroduces most of A's host-operability burden while pretending it doesn't.

**Strongest point:**
"A cron edit is a `crontab -e` with no reviewer and no diff. A workflow schedule change is a PR." Over a three-year horizon this is the single most important operability property in either proposal. Traceability of *why the schedule is what it is* is worth more than scheduler precision.

**Weakest assumption:**
"Your org already operates a CI platform... with scheduled-trigger support and secret management." B flags this honestly, but if it's false, the entire recommendation inverts — you'd be adopting a CI platform to host one cron job, which is worse than A. Everything B claims comes "for free" is free only because the platform already exists and someone else patches it.

**Hidden assumptions:**
1. That the CI runner can *reach* all the repos and any fleet endpoints. B mentions self-hosted runners for privilege, but treats network reachability as a solved detail. If a self-hosted runner is required, you now own a host with runner-agent lifecycle, disk, and patching — i.e., you've rebuilt A's utility VM but with a CI agent bolted on and *less* control over the execution environment.
2. That CI concurrency defaults will protect you from overlapping runs. B never mentions `concurrency:` groups or run-in-progress cancellation. It inherits A's exact race risk (duplicate PRs, worktree collisions) without A's explicit lock.

**Overlooked risks:**
- **Silent schedule disablement is listed but underrated as an indictment of the mechanism itself.** B rates it "high likelihood, high impact" and then mitigates with external liveness monitoring — which means the CI platform is *not* actually self-sufficient for this job. That external monitor is now un-owned infrastructure neither proposal fully specifies.
- **Concurrency control** — as above, entirely absent.
- **Vendor lock-in of the schedule and secret plumbing**, acknowledged for logic but not for the operational glue (OIDC trust config, runner labels, alert routing) which is far stickier than "keep the logic in a script" implies.

---

# Revised Recommendation

**I update toward Proposal B's mechanism, conditioned on B's stated platform assumption holding, and I fold in A's concurrency and workspace discipline as non-negotiable requirements.**

What caused the update: my original lens is "what does on-call face at 2am," and on that exact metric B wins cleanly. Host-local journald logs behind an SSH gate are a worse 2am experience than a run-history UI with per-step logs and a one-click re-run. Schedule-as-code with PR review is strictly better than host-resident unit files for the three-year, staff-turnover horizon I care about most. A's own reasoning ("do not rely on user crontabs, shared boxes, manually maintained local checkouts") is the argument *against A's own host*, extended one step further than A was willing to go: a single-purpose utility VM is a shared box that will rot.

But B is only correct because it assumes an existing CI platform. So the decision is conditional:

**If the org already operates a CI platform with scheduled triggers and secrets (B's assumption):**
Adopt B. Then bolt on A's discipline, because B omits it:
1. **Mandatory `concurrency:` group / lock** to prevent overlapping runs — this is A's strongest point and B ignores it. Duplicate cross-repo PRs are a real blast-radius event.
2. **External liveness monitor** ("alert if no successful run in N+1 intervals") owned explicitly by the same team, treated as a first-class piece of infra — because scheduled-workflow disablement makes CI not self-sufficient here.
3. **PRs-not-pushes by default, with a per-run cap on repos changed** — lifted directly from A, and the single best blast-radius control either proposal offers.
4. **Logic in a portable script, YAML only as the trigger** — caps vendor lock-in.

**If the org does *not* have a reliable scheduled-CI platform:**
Then B's "free" benefits aren't free, and A becomes correct — but only with two additions A left implicit: (a) the host configuration (unit files, flock, timer) must live in config management / IaC, not just on the box, so there is a reviewed diff and a rebuild path; and (b) an off-host log shipper and external liveness alert, so 2am debugging does not require SSH to a surviving host.

**Where I will not split the difference:** I reject running a self-hosted CI runner as a *default*. B treats it as a minor footnote, but a required self-hosted runner reconstructs A's utility-VM operability burden while giving you *less* control over the execution environment than A's plain systemd host. If you genuinely need privileged fleet network access that only a self-hosted runner can provide, the operability math tilts back toward A's explicit, boring host — because at that point you own the machine either way, and A's version is more transparent to debug than a CI agent's opaque execution wrapper.

The deciding factor is not scheduler elegance. It is: **can on-call see the last run and re-trigger it without SSH, and can the next engineer discover why the schedule is what it is via a diff?** B delivers both for free when the platform exists; A delivers neither without extra work it didn't scope.


### Openai (gpt-5.6-terra)

## Proposal A

### Steelman first

Proposal A is exactly right if the organization already has a well-managed, always-on utility VM or self-hosted runner; the job genuinely benefits from a persistent local workspace; and the team has no materially better existing CI scheduling and observability capability. In that environment, a `systemd` timer is not “just cron”: it provides durable missed-run behavior, service lifecycle controls, journal integration, resource limits, locking, and straightforward local debugging.

It is especially strong if the fleet run needs private-network access, unusual local tooling, large warm caches, long-lived checkouts, or execution characteristics that are awkward or expensive in hosted CI. For a single serial job, a managed host plus a versioned script can be the smallest viable system.

### Your assessment

**Partially agree.**

I agree with the execution-host recommendation: do not use an engineer laptop, personal credentials, or an uncontrolled shared machine. A dedicated managed runner and a portable, versioned control script are sensible.

I disagree with making `systemd` the primary scheduling and operational interface by default. It makes the host responsible for scheduling, run history, retries, notification behavior, and operator interaction. Those are exactly the concerns an existing CI platform already solves better for a team of ordinary application engineers. The proposal correctly improves on cron, but “better than cron” is not the relevant bar if CI already exists.

A managed local runner can still be used—just registered as a CI runner. That preserves local network access and durable tooling while moving scheduling, logs, manual invocation, auditability, and visibility to the system the team already uses.

### Strongest point

**It correctly prioritizes exclusive execution and blast-radius control over premature scale.**

The recommendation to use a lock, explicit timeouts, reviewable PRs, and no automatic fleet-wide direct pushes addresses the real operational danger: duplicate or unsafe changes across many repositories. That is more important than making the scheduler technically sophisticated.

### Weakest assumption

**That a durable shared workspace is necessary enough to justify host-native orchestration.**

Most cross-repository jobs can use a fresh workspace on a self-hosted CI runner, with explicitly managed caches where needed. If persistence is only a performance optimization rather than a correctness requirement, then it should not dictate the scheduling architecture.

If this assumption is false, the team inherits SSH-centric investigation, host-specific run state, and custom monitoring without gaining a meaningful capability.

### Hidden assumptions

1. **Someone owns the VM as production infrastructure.**  
   Patching, disk monitoring, image rebuilding, SSH access controls, service-account rotation, backups where relevant, and incident response do not happen automatically because the machine is called a “utility runner.”

2. **The team will actually build and maintain the missing control-plane features.**  
   The proposal calls for alerts, summaries, no-success detection, credential-expiry monitoring, and documented recovery. Those are all correct—but they are custom integrations that CI already provides partially or fully.

### Overlooked risks

- **Host compromise has unusually high blast radius.** A persistent machine with fleet-wide repository credentials, writable worktrees, and network access is an attractive target. The proposal mentions least privilege but not hardening execution against malicious repository content. A baseline job that checks out many repositories may execute untrusted build/configuration content unless deliberately designed not to.
- **Workspace contamination can create non-reproducible results.** Untracked files, stale branches, partially completed runs, cached credentials, or changed local tool state can make a run pass or fail differently from a clean execution.
- **`Persistent=true` can run at the wrong operational time.** Catch-up behavior after a prolonged outage may launch a large fleet-changing job during an incident, maintenance window, or credential-recovery event. Catch-up needs a freshness/window policy, not merely persistence.
- **Manual recovery is weak.** `systemctl start` is fine for the host owner, but it is not the same as a discoverable, audited, parameterized rerun interface available to the on-call team.
- **Journal retention is not artifact retention.** Unless explicitly exported, logs and generated reports may disappear during host rotation or disk-pressure cleanup.

---

## Proposal B

### Steelman first

Proposal B is exactly right if the organization already runs CI reliably, can provision a scoped machine identity for cross-repository access, and the task is one independently scheduled sweep rather than a dependency-heavy workflow graph. The job logic remains a normal script in a dedicated repository; CI supplies the scheduling, run history, logs, manual dispatch, retention, approvals, and notification plumbing.

If local/private-network access or nonstandard tooling is needed, a self-hosted CI runner retains the necessary execution environment without giving up CI’s operational interface. For a small team, this is the best use of existing infrastructure: one workflow and one runner, not a new orchestration service and not a custom host-operating model.

### Your assessment

**Agree.**

This is the right default mechanism: **scheduled CI, backed by a managed self-hosted runner if locality, network access, duration, or tooling requires it.**

The proposal gets the build-and-run economics right. The team should not build a scheduler, a run-history UI, a manual-retry mechanism, a log-retention system, and a notification integration on a VM when it already pays for CI. The distinction matters: a self-hosted runner is an execution environment; CI is the control plane. We need both only where local access requires it.

I would strengthen one part of B: do not assume CI’s native schedule is itself sufficient for reliability. The external “no successful run within interval” alert is mandatory, not merely a mitigation for one vendor quirk.

### Strongest point

**CI makes failure and recovery visible to the actual on-call team rather than to whoever knows how to SSH into a particular host.**

A run history, logs, code-reviewed schedule changes, manual dispatch, and retriable execution are not conveniences. They are what make this automatable by a small team over time.

### Weakest assumption

**That the CI platform and runner model can safely support the needed cross-repository privileges and network placement.**

This may fail if the job needs access to private internal services, a restricted network segment, sensitive credentials, or repository permissions that cannot be safely exposed to a broadly editable CI workflow.

If it fails, forcing the job onto ordinary shared CI creates a serious security problem: anyone able to alter the workflow or a trusted dependency may gain the fleet credential. The answer is not automatically a systemd timer, however; it is a hardened, dedicated self-hosted runner and protected CI execution path.

### Hidden assumptions

1. **The workflow repository and its execution path are protected.**  
   A fleet-wide credential in CI is safe only if workflow changes require appropriate review, protected branches prevent bypass, secrets are unavailable to untrusted pull-request contexts, and the job runs only trusted revisions.

2. **Self-hosted runner hygiene is adequate.**  
   If the runner is persistent, it needs cleanup between runs, restricted shell access, controlled runner registration, patching, disk monitoring, and a rebuild process. CI does not magically make a self-hosted host safe.

### Overlooked risks

- **Concurrent scheduled and manual runs.** The workflow needs a CI concurrency group with a deliberate policy—normally queue one or cancel the older run—not merely a lock inside the script.
- **Unsafe workflow-trigger semantics.** The job must run trusted default-branch code, not arbitrary pull-request code, and must not expose write credentials to fork or untrusted-trigger contexts.
- **Repository-content execution.** Checking out repositories is usually necessary; executing repository scripts, package hooks, or build steps under a fleet credential may not be. The baseline tool should minimize execution of repository-controlled code.
- **API, PR, and rate-limit exhaustion.** A fleet run can create too many PRs, overwhelm notification channels, hit provider API limits, or repeatedly reopen equivalent changes. It needs caps, idempotency keys, and clear behavior for existing PRs.
- **Schedule timing is best-effort.** Native CI cron may be delayed or occasionally skipped. The job must be idempotent and the external liveness monitor must alert on absence of completion, not merely failed runs.
- **Runner persistence can still cause contamination.** If using a self-hosted runner, use per-run clean workspaces or explicit cleanup; do not treat the CI wrapper as proof of reproducibility.

---

# Revised recommendation

**I recommend Proposal B’s mechanism: a scheduled CI workflow in a dedicated `fleet-baseline` repository, using a dedicated managed self-hosted runner where local/private-network access or specialized tooling requires it.**

This is not a compromise between A and B. The scheduler and operator interface should be CI. The execution host may be local and persistent, but it should not own the schedule.

The deciding argument is operational ownership: a team small enough to avoid Airflow, Kubernetes, and a bespoke orchestrator is also too small to voluntarily recreate CI’s run history, manual dispatch, audited schedule changes, artifact retention, and failure visibility around a `systemd` timer. Proposal A correctly identifies the need for a managed runner, locking, timeouts, and safe PR-based remediation. Those controls should be implemented **inside a CI-governed job**, not as a substitute for CI.

## Concrete implementation

1. Create a dedicated `fleet-baseline` control repository:
   - portable orchestration CLI/script;
   - explicit repository manifest;
   - pinned tool versions;
   - schedule-as-code;
   - runbook and ownership declaration.

2. Run it through scheduled CI plus manual dispatch:
   - configure a CI concurrency group to prevent overlap;
   - set explicit timeout and cancellation behavior;
   - retain logs and summary artifacts for at least 30 days;
   - post concise success/failure summaries to the existing operational channel.

3. Use a dedicated self-hosted runner only if needed:
   - isolated runner label available only to this workflow;
   - no interactive general-purpose use;
   - clean workspace per run;
   - monitored disk, patching, credential rotation, and documented rebuild;
   - private-network placement only where required.

4. Secure the fleet credential:
   - scoped service identity, preferably short-lived tokens;
   - protected default branch and required review for workflow changes;
   - no credentials in untrusted PR contexts;
   - avoid executing repository-controlled scripts with fleet-write authority.

5. Make the job safe by default:
   - generate reports or reviewable PRs, not direct fleet-wide pushes;
   - cap repositories/PRs changed per run;
   - make reruns idempotent;
   - alert externally if no successful run occurs within 1.5–2 expected intervals.

6. Revisit only when the workload proves it needs more:
   - move to an orchestrator for multiple dependent schedules, backfills, or real SLAs;
   - shard only when run duration materially consumes the interval or API limits demand it;
   - consider a separate hardened execution platform only when privilege/network requirements exceed what a protected self-hosted CI runner can safely provide.


## Synthesis (by gemini, non-participant)

## Consensus
Both participants reached a strong, evidence-backed consensus by the second round, driven by rigorous cross-examination rather than groupthink. 
* **Avoid heavy orchestrators:** Do not adopt Airflow, Temporal, K8s `CronJob`, or Nomad for a single recurring job.
* **Control plane:** Scheduled CI (GitHub Actions / GitLab CI) is vastly superior to host-native scheduling (cron or `systemd`) due to built-in run history, log retention, schedule-as-code (PRs with diffs), and manual re-trigger capabilities (`workflow_dispatch`).
* **Blast-radius control:** The job must default to generating pull requests (or reports) rather than direct pushes, and concurrency locking is mandatory to prevent overlapping runs from spamming duplicate PRs or racing on shared states. 
* **Silent failure mitigation:** CI-native scheduling is best-effort and often disabled after repository inactivity. An external liveness monitor (alerting if no successful run occurs in $N+1$ intervals) is strictly required.

## Unresolved Disagreements
Because both participants converged on the CI control-plane model, the remaining disagreement centered on the execution environment.
* **The Crux:** Should a persistent, self-hosted runner be the *default* execution environment to provide a warm workspace and local tooling, or should ephemeral hosted CI runners be the default to minimize operational burden?
* **Stronger Argument:** The argument for **ephemeral hosted CI runners by default** (unless network/privilege constraints forbid it) was stronger. The counter-argument heavily underestimated the "pet VM" operability burden—patching, disk exhaustion from stale worktrees, and ownership decay. Ephemeral runners force clean execution and prevent host-rot, which is paramount for an automation task designed to reduce toil.

## Argument Quality Assessment
Both participants delivered excellent, well-reasoned arguments that adapted based on the opponent's valid points. 
* **Strongest Argument:** The observation that *"a cron edit is a `crontab -e` with no reviewer and no diff, while a workflow schedule change is a PR,"* combined with the "2am on-call lens" (the ability to see a red/green status and logs without SSHing into a box). This decisively killed the `systemd` utility VM proposal.
* **Weakest Argument:** The initial claim that a `systemd` utility VM is "small" and requires "minimal new infrastructure." A single, persistent, snowflake VM that requires custom log shipping, manual failure alerting, and OS patching is exactly the type of technical debt that rots over a 3-year horizon.
* **Best Synthesizer:** Both participants expertly synthesized the other's points in Round 2, though the argument to map the host-native `flock` concept directly onto CI's `concurrency:` groups was a particularly sharp refinement.

## Blind Spots
* **API Rate Limits:** Running a cross-repo sweep implies interacting with tens or hundreds of repositories. Neither participant deeply examined the risk of hitting GitHub/GitLab API rate limits during cloning or PR generation, which is a common failure mode for fleet-wide automation.
* **Malicious Code Execution:** Checking out hundreds of repositories under a highly privileged token carries supply-chain risks. If the baseline tool executes arbitrary repository content (e.g., executing `make` or running local hooks), a compromised downstream repo could exfiltrate the fleet-wide credential.
* **State Management:** The debate didn't address how the script tracks previously opened PRs. If it queries the API for every repo on every run, it will be slow and fragile. If it relies on local state, ephemeral runners will break it.

## Recommended Decision
**Host the recurring run via a scheduled CI pipeline in a dedicated `fleet-baseline` repository, using ephemeral hosted runners by default.** 

The scheduler, logs, and manual operator interface should live entirely in the CI platform to guarantee 2am debuggability and schedule-as-code traceability. 

If—and only if—the job strictly requires access to a private internal network or specific fleet-admin credentials that cannot safely leave your infrastructure, use a self-hosted CI runner. However, treat the self-hosted runner strictly as a dumb execution node, retaining the CI platform as the control plane. Enforce strict concurrency controls within the CI pipeline to prevent overlapping runs.

## Risks
* **Silent Schedule Disablement:** CI platforms (like GitHub Actions) routinely disable scheduled workflows on inactive repositories. *Mitigation:* Implement an external "dead man's switch" / liveness monitor (e.g., Datadog or PagerDuty cron integration) that alerts the team if a successful run hasn't occurred within the expected time window.
* **Blast Radius & PR Spam:** A bug in the baseline script could open hundreds of bad PRs across the organization simultaneously. *Mitigation:* Implement a hard-coded cap on the number of repositories mutated/PRs opened per run. 
* **Credential Sprawl:** A token with write access to all repos is a high-value target. *Mitigation:* Use short-lived, scoped OIDC tokens (e.g., GitHub Apps) rather than static Personal Access Tokens. Ensure the CI workflow runs on a protected default branch and secrets are not exposed to untrusted pull requests.

## Action Items
1. Create a new `fleet-baseline` repository to house the baseline script and the CI workflow YAML.
2. Write the baseline logic in a portable language/script (e.g., Python or Bash) so it can be run locally for testing, keeping the CI YAML purely for orchestration.
3. Configure the CI workflow with both a `schedule` trigger and a `workflow_dispatch` trigger (for manual 2am re-runs).
4. Add a `concurrency` group definition to the CI workflow to ensure only one instance of the baseline runs at a time (canceling or queueing pending runs).
5. Provision a scoped machine identity (e.g., a GitHub App) with the minimum required permissions, and map it to the CI workflow via OIDC.
6. Set up an external liveness monitor in your alerting stack to page if the workflow stops reporting successes. 
7. Enforce branch protection on the `fleet-baseline` repo to require code review for any schedule or logic changes.
