# AI Council Debate: For a small Python library's dependency policy, pick one: a committed lockfile, 

**Panel:** claude-opus-4-8, gpt-5.6-terra
**Synthesizer:** gemini (non-participant)
**Rounds:** 2
**Panel Mode:** custom
**Debate Mode:** pick
**Source:** cli

## Verdict Summary

**Decision:** Choose floating minimum-version ranges for the published library.
**Dissent:** non-unanimous (see minority report)
**Panel seated:** 2/2
**Verdict author:** gemini (non-participant)

_Machine-readable fields are authoritative in the council-verdict-*.json sibling._

---

## Question

For a small Python library's dependency policy, pick one: a committed lockfile, or floating minimum-version ranges.

## Round 1: Initial Responses

### Claude (claude-opus-4-8)

## Position
For a **library**, use floating minimum-version ranges in your package metadata, and add a committed lockfile *only* for your dev/CI environment. If forced to pick exactly one for the published package, choose **floating minimum-version ranges**.

## Alternatives Considered
- **Committed lockfile as the dependency policy (the published constraint):** Rejected. A library doesn't own the runtime environment — the consuming application does. Pinning exact versions in a library forces conflicts on every downstream consumer and makes your package uninstallable alongside anything with overlapping dependencies. Lockfiles solve reproducibility for *deployables*, not for *reusable components*.
- **Wide-open ranges with no lower bound (`requests`, no `>=`):** Rejected. This is what "floating" degrades into if done lazily. It silently allows an ancient version that lacks a function you call, producing an `AttributeError` in a stranger's production, not yours.

## Decision Criteria
1. **Who controls the runtime** — library authors don't; application authors do. This is the deciding constraint.
2. **Downstream co-installability** — the policy must not make the package a resolver landmine.
3. **Reproducibility of your own CI/development** — you still need to detect breakage deterministically.
4. **Maintenance cost over 3 years** — how often will a maintainer be forced to touch this.
5. **Failure clarity at 2am** — when a consumer hits an incompatibility, the error should point at a version boundary, not a mystery.

## Assumptions
1. This is genuinely a *library* meant to be imported by other projects, not an application or CLI deployed as a unit. (If it's actually an app, my answer flips to lockfile.)
2. You have CI capable of running against a resolved dependency set.
3. You can afford minimal maintenance attention (occasional range bumps), but not constant lockfile-churn PRs.

## Recommendation
**Now:**
- In `pyproject.toml`, declare **minimum-version lower bounds** with a rationale: `>=X.Y` where `X.Y` is the oldest version whose API you actually rely on.
- Add **upper caps only where a dependency has a history of breaking SemVer** (e.g., `pydantic<3`), not reflexively.
- Commit a **lockfile for CI/dev only** (`uv.lock`, `requirements-dev.lock`, or `poetry.lock`) so *your* test runs are reproducible.
- Run CI on **two axes**: the locked set (reproducible) and a "lowest-supported" resolution (`uv pip install --resolution=lowest-direct` or `pip install` against pinned floors) to prove your minimums are honest.

**Not now:**
- Don't ship the lockfile as the installed constraint.
- Don't add blanket upper caps on every dependency — caps are the biggest cause of unresolvable downstream environments.

**Defer:**
- Tightening minimums until a consumer or a new API usage forces it.
- Adding a compatibility test matrix across Python versions unless you actually support several.

## Rationale
- A committed lockfile in a published library propagates a single exact version graph into every consumer's resolver — the classic cause of `ResolutionImpossible`. Floating ranges are the only policy that lets N applications share your library.
- Minimum bounds encode a real contract: "I call `feature()` added in 2.3, so `>=2.3`." That's an explicit, greppable statement of what you depend on — exactly the kind of contract that survives a maintainer handoff.
- The "lockfile for CI, ranges for publish" split gives you **both** reproducible testing and downstream flexibility — it's not either/or; the false dilemma is in the question.
- Testing the *lowest* resolution catches the most common silent bug: bumping to a new API locally while your published floor still claims to support an older version that lacks it.
- Upper caps are debt: they age into forced-conflict bugs the moment a downstream needs the newer version. Cap only against dependencies with a demonstrated SemVer-violation record.

## Trade-offs
- You give up guaranteed byte-identical dependency resolution for *end users* (they'll get whatever their resolver picks) — acceptable, because that's the consumer's job to lock.
- Slightly more CI complexity (two resolution runs instead of one) and periodic manual review of whether floors are still accurate.
- You accept the small risk of a transitive dependency shipping a bad release — mitigated by the consumer's lockfile, not yours.

## Risks
- **Stale/incorrect lower bounds** (likelihood: high over 3 years; impact: medium) — you use a newer API but never raised the floor, breaking older-version consumers. *Mitigate:* the lowest-resolution CI job catches this on the PR that introduces it.
- **A dependency breaks SemVer without a cap** (likelihood: medium; impact: medium) — new major/minor breaks your users. *Mitigate:* add a targeted cap reactively; monitor dependency changelogs for your handful of directs.
- **Reflexive over-capping creep** (likelihood: medium; impact: high downstream) — maintainers add `<next-major` everywhere out of caution, making the package hard to co-install. *Mitigate:* document a caps policy — caps require a cited breakage, not a hunch.

## Signals to Revisit
- The artifact stops being a library and becomes a deployed service/app → switch to lockfile-as-truth.
- You accumulate frequent downstream bug reports about incompatible transitive versions → tighten bounds or add targeted caps.
- A dependency you rely on begins breaking SemVer repeatedly → introduce upper caps for that specific package.
- Maintenance burden of manual floor-bumping becomes noticeable → automate lower-bound verification (e.g., a bot that raises floors when new APIs are used).


### Openai (gpt-5.6-terra)

## Position

Use floating minimum-version ranges for the library; do not commit a runtime lockfile.

## Alternatives Considered

- Committed lockfile: rejected because it primarily pins an application’s deployable environment, while a library must remain compatible with a range of consumer environments.
- Unbounded dependencies: rejected because reproducibility and timely detection of upstream breaking changes still require declared lower bounds and controlled CI environments.

## Decision Criteria

- Small team’s ability to maintain the policy.
- Compatibility with downstream users’ dependency resolvers.
- Reproducible development, CI, and releases.
- Early detection of upstream breaking changes.
- Minimal release and operational overhead.

## Assumptions

- This package is a reusable library, not an end-user application or service.
- Dependencies follow semantic versioning reasonably well.
- The team can run CI on a small dependency-version matrix.
- The project has few direct runtime dependencies.

## Recommendation

Declare bounded minimum-version ranges in package metadata, e.g. `requests>=2.28,<3`. Test the declared minimum supported versions and current latest compatible versions in CI.

Do not commit a runtime lockfile as the library’s dependency contract. Do not use unconstrained `>=` ranges where the next major version may be incompatible.

Defer a committed lockfile to development tooling only if needed for reproducible contributor/CI environments; keep it separate from published library dependencies.

## Rationale

- Consumers need their resolver to select versions compatible with their wider application, which a library lockfile cannot safely dictate.
- A small team can maintain 2–3 CI environments per Python version more easily than continuously updating and validating a lockfile.
- Upper major-version bounds prevent surprise breaking upgrades while avoiding unnecessary pins.
- Testing both minimum and latest compatible versions catches “declared but untested” support gaps.

## Trade-offs

- Builds are less byte-for-byte reproducible than an application using a lockfile.
- CI needs a small matrix, typically 2 dependency sets × supported Python versions.
- An upstream patch/minor release can expose an incompatibility before the library team updates constraints.

## Risks

- **Minimum versions silently stop working** — likelihood: medium; impact: medium. Mitigate with a CI job that installs lowest supported versions.
- **Upstream major release breaks users** — likelihood: low-to-medium; impact: high. Mitigate with `<next-major` bounds and dependabot/renovate-style update checks.
- **Security remediation requires an urgent floor increase** — likelihood: low; impact: medium. Mitigate with prompt patch releases raising the minimum version.
- **Contributor environments drift** — likelihood: medium; impact: low. Mitigate with an optional development lockfile or pinned CI requirements.

## Signals to Revisit

- The project becomes an application, CLI distribution, or deployed service.
- Reproducibility or supply-chain controls become a contractual requirement.
- The dependency graph becomes large or includes historically unstable packages.
- CI data shows frequent failures from compatible upstream minor releases.


## Round 2: Critique

### Claude (claude-opus-4-8)

# Proposal A

**Steelman first.** This is exactly right when the library has *few direct dependencies that follow SemVer honestly*, a maintainer who will actually run the lowest-version CI job, and consumers whose resolvers must retain freedom to co-install. Under those conditions, `>=X,<next-major` plus a two-point CI matrix (floor + latest) is the minimum viable contract that survives a handoff: greppable bounds, no lockfile churn, no resolver landmines.

**Your assessment.** **Agree** with the Position. A published library must not dictate an exact version graph to consumers. That's not a preference, it's a property of how Python resolution works.

**Strongest point.** "Testing both minimum and latest compatible versions catches *declared but untested* support gaps." This is the operationally honest move — a floor you never install is a lie in your metadata.

**Weakest assumption.** "Dependencies follow semantic versioning reasonably well." Half the popular Python ecosystem doesn't. If this is false, the reflexive `<next-major` cap either fails to protect (breakage in a minor) or over-protects (blocks a safe major). The whole "upper major-version bounds prevent surprise breaking upgrades" claim rests on SemVer being trustworthy, and it frequently isn't.

**Hidden assumptions.** (1) That a **blanket `<next-major` cap is safe by default** — it treats caps as free insurance, but every cap is a future forced-conflict for downstream. Proposal B catches this; A does not notice it's making the assumption. (2) That the **maintainer will keep the floor honest** — A lists "minimum versions silently stop working" as a *mitigated* risk, but the mitigation (a CI job) only works if someone reads the failure and raises the floor. It assumes attentiveness it never budgets for.

**Overlooked risks.** Transitive dependency breakage — A only reasons about direct deps. A bad transitive release breaks users regardless of your direct bounds, and the answer (consumer's lockfile) is never stated. Also: **security floor-raising creates a support-window conflict** — raising the minimum to patch a CVE can strand consumers pinned below it; A flags it as low-impact but doesn't address the coordination cost.

---

# Proposal B

**Steelman first.** Right when the maintainer is disciplined enough to (a) treat caps as debt requiring a *cited* breakage, (b) actually run a lowest-*direct* resolution in CI, and (c) resist the temptation to reflexively cap. B is A plus the correction that the question is a false dilemma — lockfile for dev, ranges for publish is the real answer.

**Your assessment.** **Agree**, and this is the stronger of the two. It answers the same Position but names the trap A walks into: reflexive over-capping.

**Strongest point.** "Upper caps are debt: they age into forced-conflict bugs the moment a downstream needs the newer version. Cap only against dependencies with a demonstrated SemVer-violation record." This is the single most operationally important sentence across both proposals. Caps are the number-one cause of `ResolutionImpossible` in the wild, and B is the only one that treats them as a liability rather than a safety feature.

**Weakest assumption.** "You have CI capable of running against a resolved dependency set" *and* a lowest-direct resolution run. For a *small* library this is real setup cost, and B assumes the team will build and maintain a two-axis matrix. If they won't, the lowest-resolution job — B's entire defense against stale floors — silently doesn't exist, and B degrades into "floating ranges with unverified floors," which is worse than A because B explicitly declines the caps A would've added.

**Hidden assumptions.** (1) That **`--resolution=lowest-direct` reflects what consumers actually get** — it doesn't test lowest *transitive*, so a floor can pass B's CI and still break a consumer whose resolver pulls an old transitive. B presents lowest-direct as proof the minimums are honest; it's proof only for directs. (2) That the **maintainer distinguishes "SemVer-violation record" reliably** — this requires judgment and changelog-watching that a solo maintainer under time pressure won't consistently apply. B's caps policy is only as good as the person enforcing it.

**Overlooked risks.** **Lockfile rot in dev/CI** — a committed dev lockfile that no one updates means CI passes against a fossilized graph while real consumers install fresh versions; the reproducibility becomes a false-comfort trap. B never addresses staleness of its *own* lockfile. Also: **the two-axis CI can green-light a broken middle** — passing at floor and at latest-locked says nothing about the versions in between, which is what most consumers actually resolve.

---

# Revised Recommendation

**I maintain the position both proposals share, and adopt Proposal B's framing over A's.** Floating minimum-version ranges for the published package; committed lockfile for dev/CI only. The question's either/or is false and I won't pretend otherwise.

**What moved me:** B's caps-as-debt argument. Going in, I'd have defaulted to A's reflexive `<next-major` as cheap insurance. B is correct that a cap is a *contract you impose on strangers' resolvers*, and an explicit contract that ages into a forced conflict is worse than no cap — it fails at 2am in someone else's production with a `ResolutionImpossible` they can't fix without forking you. That reframing changed my default from "cap by default, unpin when annoying" to "**don't cap unless you can cite the breakage**."

**Where I go further than both** — the 2am test exposes what neither fully closes:

1. **The floor is the load-bearing contract, and both under-defend it.** A's floor-CI and B's lowest-direct both fail to test lowest *transitive*. For a small library, add a `--resolution=lowest` (full, not just direct) job. It's one line and it's the only thing that makes the floor an honest promise rather than an aspiration.

2. **A committed dev lockfile is a liability if it rots.** Either wire it to auto-update (renovate/dependabot on the lockfile) or don't commit it. A green CI against a fossil is worse than no lockfile because it manufactures false confidence — the exact thing the on-call engineer trusts and gets burned by.

3. **Every bound needs a cited reason in the metadata**, not just a number. `requests>=2.28  # need JSONDecodeError, added 2.28` and `pydantic<3  # v2 dropped .dict(), see #142`. This is the handoff-survival property: three years out, a maintainer can *grep the reason* and know whether the bound is still true. A number alone is undecidable — no one dares touch it, so it ossifies. That's the cognitive-load debt that actually kills libraries.

**Net:** B's position, plus lowest-*transitive* CI, plus a no-rot rule for the dev lockfile, plus mandatory cited rationale on every bound.


### Openai (gpt-5.6-terra)

### Proposal A

**Steelman:** Exactly right if the library has few stable direct dependencies, supports several Python versions, and the team can reliably run minimum and latest-compatible CI jobs.

**Assessment:** Partially agree. Floating ranges are the right choice, but blanket `<next-major` caps create downstream resolver conflicts and ongoing maintenance work that a small team may not sustain.

**Strongest point:** A library must preserve downstream dependency-resolution flexibility; a runtime lockfile is not its compatibility contract.

**Weakest assumption:** That upper major-version caps are cheap insurance. They often become stale and block consumers that need a newer major version.

**Hidden assumptions:**

- The team can maintain a dependency matrix across every supported Python version.
- Dependencies follow SemVer closely enough that “latest compatible” is a meaningful safety boundary.

**Overlooked risks:**

- Universal upper bounds can make otherwise-compatible libraries impossible to install together.
- A small team may let the matrix rot, producing nominal rather than real compatibility support.

### Proposal B

**Steelman:** Exactly right if “dependency policy” distinguishes published runtime constraints from the project’s own development/CI environment, and the team can support two resolution modes in CI.

**Assessment:** Agree with its position: choose floating minimum-version ranges for the library. However, its “add a dev/CI lockfile” recommendation is outside the forced either/or decision and should not obscure the actual choice.

**Strongest point:** The runtime environment belongs to the consuming application, so the application—not the library—must lock the full graph.

**Weakest assumption:** That the team can afford and correctly configure lowest-version CI. If it does not, lower bounds become unverified promises and consumers on old supported versions will break.

**Hidden assumptions:**

- Consumers use competent resolvers and lock their own deployments.
- Direct lower bounds adequately constrain problematic transitive dependencies.

**Overlooked risks:**

- A compromised or withdrawn upstream release can affect fresh installs even when declared ranges are correct.
- “Targeted caps only after breakage” can leave early adopters exposed to a known-incompatible release before the library reacts.

## Revised recommendation

Choose **floating minimum-version ranges** in published metadata.

Use explicit lower bounds derived from APIs actually used, e.g. `requests>=2.28`. Do **not** commit a runtime lockfile as the library’s dependency policy. Add upper bounds only for demonstrated incompatibilities or dependencies with a concrete compatibility boundary—not as a default.

The proposals reinforce rather than change this position: A correctly prioritizes downstream compatibility but overestimates the practicality of blanket caps; B correctly identifies ownership of the runtime graph and the operationally realistic path for a small library team.


## Synthesis (by gemini, non-participant)

## Consensus
Both participants overwhelmingly agree: **A library must use floating minimum-version ranges for its published metadata.** Both strongly reject using a committed lockfile as the published dependency policy. 

This consensus was driven by strong shared evidence regarding how Python package resolution works: because the consuming application owns the runtime environment, a library that dictates exact versions (via a lockfile) will inevitably cause `ResolutionImpossible` conflicts when co-installed with other libraries.

Additionally, both participants arrived at a consensus against the use of "blanket" upper bounds (e.g., `<3.0.0`). While OpenAI initially advocated for `<next-major` caps, Claude's rebuttal demonstrated that these act as resolver landmines, prompting OpenAI to drop the recommendation.

## Unresolved Disagreements
By the end of the debate, there were no fundamental disagreements on the core technical approach, but there was a minor divergence on framing:
- **The Crux:** Strict adherence to the prompt vs. holistic operational reality. OpenAI noted that recommending a dev-only lockfile falls outside the prompt's forced either/or choice. Claude insisted the prompt presented a "false dilemma" and that separating published ranges from dev lockfiles is the only logically sound answer.
- **Stronger Argument:** Claude's framing is stronger. Recommending floating ranges without acknowledging the necessity of a locked development environment ignores how modern Python CI/CD functions. You cannot responsibly maintain floating ranges without a locked baseline for development.

## Argument Quality Assessment
- **Best-reasoned proposal:** Claude's arguments were consistently exceptional, properly distinguishing between the needs of the consumer (flexibility) and the developer (reproducibility).
- **Single strongest argument:** Claude's assertion that upper caps are "debt" and "a contract you impose on strangers' resolvers." Framing upper bounds as eventual forced-conflict bugs that break strangers' production environments—unless backed by cited, historical breakage—was the most insightful point in the debate and successfully changed OpenAI's stance.
- **Single weakest argument:** OpenAI’s initial Round 1 assumption that dependencies follow semantic versioning reliably enough to make `<next-major` a safe default. The Python ecosystem notoriously struggles with SemVer, making proactive capping highly disruptive.

## Blind Spots
- **Transitive Dependency Vulnerabilities:** The debate focused heavily on direct dependencies. If a transitive dependency has a severe CVE, a library relying purely on floating direct bounds has limited tools to force an upgrade for its consumers, aside from artificially adopting the transitive dependency as a direct one just to bump its floor.
- **Overhead for a "Small" Library:** Both models recommended a dual-axis CI setup (testing latest versions and minimum versions). For a truly *small* library, configuring and maintaining a matrix that tests `--resolution=lowest` across multiple Python versions represents a heavy upfront boilerplate tax that neither participant adequately weighed against the "small library" constraint.

## Recommended Decision
**Choose floating minimum-version ranges for the published library.**

**Rationale:** 
Libraries are building blocks, not deployable apps. If a library publishes a lockfile, it forces its exact dependency tree onto the consumer. If two libraries do this, and share a dependency but disagree on the patch version, the consumer's application cannot be installed. Floating minimum ranges allow the consumer's package manager to calculate a mutually agreeable dependency graph.

Upper caps should **not** be used by default. They should only be introduced reactively when a dependency proves to be incompatible. 

*Important operational caveat:* While the published package must use floating ranges, you must still generate and commit a lockfile strictly for your own development and CI environments. This ensures your contributors have reproducible builds.

## Risks
1. **Stale/Dishonest Minimums:** A maintainer uses a new feature from a dependency (e.g., a function added in v2.5) but leaves the metadata at `>=2.0`. Consumers on v2.1 will install the library successfully, but experience runtime crashes.
   - *Mitigation:* Implement a CI job that strictly installs the lowest declared versions.
2. **Upstream Breakages:** A dependency releases a breaking change that breaks your library. Because you have no upper cap, users doing fresh installs pull the broken version.
   - *Mitigation:* React quickly by issuing a patch release that adds a targeted upper cap for that specific dependency. Rely on the fact that responsible consumers should have their own lockfiles protecting their production environments.
3. **Dev Lockfile Rot:** The development lockfile gets ignored, meaning CI runs against ancient dependency versions that no real user is currently resolving.
   - *Mitigation:* Automate dev lockfile updates using a dependency bot.

## Action Items
1. Define dependencies in `pyproject.toml` (or equivalent) using explicit lower bounds (e.g., `requests>=2.28.0`). Do not use wide-open bounds (e.g., `requests`).
2. Add an inline comment to every lower bound explaining *why* that version was chosen (e.g., `requests>=2.28 # requires JSONDecodeError`). This ensures future maintainers can safely audit the floors.
3. Remove any default `<next-major` upper bounds. Only apply upper bounds if you can point to a specific, known incompatibility.
4. Configure two CI testing jobs:
   - **Latest:** Standard install, resolving the newest compatible versions.
   - **Lowest:** Install using minimum bounds (e.g., using `uv pip install --resolution=lowest-direct` or `pip install` against a constrained floor list) to verify your lower bounds are honest.
5. Generate a lockfile (`uv.lock`, `poetry.lock`, etc.) and commit it **exclusively** for development and the "Latest" CI run to ensure contributor reproducibility. Set up Dependabot/Renovate to keep this file fresh.
