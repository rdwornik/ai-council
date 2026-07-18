# AI Council Debate: What is the canonical filename / foldername separator across the Dev ecosystem?

**Panel:** claude-opus-4-8, gpt-5.6-terra
**Synthesizer:** gemini (non-participant)
**Rounds:** 2
**Panel Mode:** custom
**Debate Mode:** pick
**Source:** cli

## Verdict Summary

**Decision:** Adopt **lowercase kebab-case (`-`)** as the canonical default separator for all new, general-purpose files and directories.
**Dissent:** non-unanimous (see minority report)
**Panel seated:** 2/2
**Verdict author:** gemini (non-participant)

_Machine-readable fields are authoritative in the council-verdict-*.json sibling._

---

## Question

What is the canonical filename / foldername separator across the Dev ecosystem?

## Round 1: Initial Responses

### Claude (claude-opus-4-8)

## Position
Use the **hyphen (`-`) as the canonical separator — kebab-case — for filenames and foldernames** across the Dev ecosystem. Reserve underscores and dots for structural/semantic roles (extensions, test/fixture suffixes, language-mandated module names), not as the general-purpose word separator.

## Alternatives Considered
- **snake_case (`_`) everywhere** — Rejected as the *canonical* choice because it collides with language-level identifier conventions (Python modules, SQL, env vars). If underscores are both "the word separator" and "the Python-import-safe separator," you lose the ability to distinguish a stylistic separator from a semantically load-bearing one. It also fights URL/DNS ergonomics.
- **No enforced convention (mixed case, spaces, camelCase files)** — Rejected outright. Spaces break shell pipelines and require quoting (a real hazard given the global rule already calls out paths with spaces/parens). CamelCase filenames are the single largest source of cross-platform breakage because Windows/macOS are case-insensitive and Linux is not.

## Decision Criteria
1. **Cross-platform safety** — must survive case-insensitive (Windows/macOS) → case-sensitive (Linux/CI) round-trips without collision.
2. **Shell & tooling ergonomics** — no quoting required; `Tab`-completion and glob-friendly.
3. **Distinct semantic channels** — the separator must not overload a character the language/toolchain already needs (`.` for extensions, `_` for identifiers).
4. **Consistency with existing precedent** — the memory system in this very ecosystem already mandates `<short-kebab-case-slug>` for memory names.
5. **URL/DNS parity** — hyphen is the web-native word separator; foldernames often become path segments.

## Assumptions
1. "Dev ecosystem" means the collection of repos, scripts, and tooling under this user's control — **not** a single language's package internals where the language dictates naming.
2. The environment is **mixed Windows + Linux CI**, so case-folding collisions are a live risk, not theoretical.
3. Files that must be importable as code modules (e.g., Python `.py`) are a bounded minority, not the majority of files being named.

## Recommendation
**Do now:**
- Adopt **kebab-case** as the default: `sync-mywork.ps1`, `learned-rules.md`, `code-review-config.json`.
- Lock filenames and foldernames to **lowercase** to eliminate case-collision risk.
- Codify the reserved roles: `.` = extension/version boundary only (`config.prod.yaml`); `_` = permitted **only** where a toolchain requires it (Python modules, `__init__.py`, test discovery patterns).

**Do not do now:**
- Do not mass-rename existing language-mandated files (Python modules, framework-convention files) — that trades a stylistic win for a breakage risk.
- Do not enforce this on generated/vendored directories (`node_modules`, `.venv`).

**Defer:**
- A lint rule / pre-commit hook enforcing the convention. Write the rule down first, live with it for a few weeks, then automate once the exceptions are known.

## Rationale
- **Case-collision is the 2am incident this prevents.** A repo with `MyWork/` and `mywork/` builds fine on the dev's Mac and detonates on Linux CI with a confusing "file not found." Lowercase-kebab removes the entire class.
- **The ecosystem already voted.** The memory subsystem mandates kebab-case slugs. Picking a *different* separator for files would create two conventions to remember — cognitive load is the tax you pay every time you name something.
- **Hyphen keeps `_` and `.` meaningful.** When `_` is reserved, seeing `test_helpers.py` tells you it's Python; seeing `test-helpers.md` tells you it's just prose. Overloading `_` destroys that signal.
- **Shell-safe by construction.** No PowerShell/Bash quoting, glob-friendly, no `parens`/spaces — directly satisfies the existing global rule about hazardous paths.
- **~90% coverage from one rule.** The only genuine exceptions are language-mandated (Python modules, some framework files); everything else — docs, scripts, configs, folders — takes the hyphen cleanly.

## Trade-offs
- You accept a **two-tier rule** ("hyphen by default, underscore where a toolchain demands it") rather than one blanket separator. That's a small memorization cost in exchange for not fighting the language.
- Kebab-case files **cannot be imported directly** in most languages, so any file destined to be a code module is an explicit exception — a boundary you must remember to draw.

## Risks
- **Silent drift back to mixed conventions** (likelihood: high without automation, impact: low-moderate). Mitigate with the deferred pre-commit hook once exceptions are catalogued.
- **Someone kebab-cases a Python module and breaks an import** (likelihood: moderate, impact: moderate). Mitigate by stating the "underscore where the toolchain requires it" rule *loudly* and up front, not as a footnote.
- **Legacy CamelCase paths lingering** cause the exact collision this policy prevents, but only if half-migrated (likelihood: moderate, impact: high). Mitigate by not doing partial renames — either leave a subtree alone or convert it wholesale.

## Signals to Revisit
- The ecosystem becomes **predominantly a single language** whose ecosystem strongly favors `_` (e.g., a Python-only monorepo) — then snake_case may earn "canonical" status and hyphen becomes the exception.
- CI moves fully to **case-insensitive storage** across the board, removing the collision driver (weakens, but doesn't reverse, the case for lowercase).
- Adoption of a tool/framework that **hard-requires** a different filename scheme for the majority of files.


### Openai (gpt-5.6-terra)

## Position

Use kebab-case: lowercase words separated by hyphens (`my-feature`, `api-client`). It is the most portable, readable, URL-friendly default across Dev tooling.

## Alternatives Considered

- `snake_case` — common in Python and some generated assets, but less consistent with web URLs, package names, and CLI conventions.
- `camelCase` — suitable for code identifiers, but less readable in paths and can cause case-sensitivity problems across operating systems.

## Decision Criteria

- Works reliably on Windows, macOS, Linux, Git, shells, and URLs.
- Is readable in terminals and file explorers.
- Avoids case-sensitivity and escaping issues.
- Aligns with common package, repository, and web-project conventions.
- Is easy for a mixed-language team to apply consistently.

## Assumptions

- This is a default convention for human-created files and folders, not a language-mandated filename.
- Existing repository conventions take precedence to avoid unnecessary churn.
- Generated files and ecosystem-required names may use their required format.

## Recommendation

Use lowercase kebab-case for new general-purpose folders and filenames: `user-profile/`, `api-client.ts`, `deployment-guide.md`.

Do not introduce spaces, mixed casing, or separator inconsistency. Do not rename established directories merely to conform.

Defer exceptions to documented cases: Python modules (`snake_case.py`), language/framework-required names, and conventional special files such as `README.md`, `Dockerfile`, and `package.json`.

## Rationale

- Hyphens are safe in common filesystems, Git paths, URLs, package names, and static-site routes.
- Lowercase avoids cross-platform collisions such as `API/` versus `api/`, which behave differently on typical Windows/macOS and Linux setups.
- Kebab-case is immediately scannable in paths: `billing-service/payment-retry-policy.md`.
- A single general rule reduces review friction; exceptions are limited to language or tool requirements.

## Trade-offs

- Python-centric teams will use two conventions: `snake_case` for Python modules/identifiers and kebab-case for general paths.
- Hyphenated names require quoting or escaping in a few shell contexts when used as command arguments ambiguously; this is uncommon for paths.
- It may differ from legacy repository conventions, where consistency should win.

## Risks

- **Risk:** Tooling expects a prescribed filename or casing. **Likelihood:** medium; **impact:** high. **Mitigation:** preserve framework and tool-required names exactly.
- **Risk:** Team applies kebab-case to Python importable modules. **Likelihood:** medium; **impact:** medium. **Mitigation:** document `snake_case` as the Python-module exception.
- **Risk:** Renaming existing paths breaks imports, scripts, or CI. **Likelihood:** medium; **impact:** high. **Mitigation:** apply the convention prospectively unless a planned migration has automated reference updates.

## Signals to Revisit

- The primary language becomes Python, where importable-module naming dominates path decisions.
- A framework or deployment platform imposes a conflicting naming convention.
- Existing repositories overwhelmingly use a different convention and migration cost exceeds the consistency benefit.


## Round 2: Critique

### Claude (claude-opus-4-8)

Both proposals converge on the same answer — lowercase kebab-case — so my job is to test whether that convergence is *sound* or just two people sharing the same blind spot. As the person on call at 2am, I care less about the winning separator and more about whether the *exceptions* are drawn where they'll actually hurt.

---

## Proposal A

**Steelman.** If the Dev ecosystem is genuinely polyglot and web-adjacent — repos that become URLs, static routes, package names — then kebab-case is the separator that survives the most contexts without escaping. A's real strength is its restraint: "apply prospectively, don't rename established directories." For that to be exactly right, the dominant cost in this ecosystem must be *review friction and cross-platform collisions on new files*, not migration of a large legacy corpus. If most naming decisions are green-field, A is correct with minimal ceremony.

**Assessment: Partially agree.** The Position is right; the articulation is under-specified where it matters most. A treats snake_case as a Python footnote but never says what the separator *means* — it lists exceptions without giving the on-call engineer a rule to *derive* new exceptions from. That's the difference between a policy and a list.

**Strongest point.** The case-collision argument: `API/` vs `api/` behaving differently on Windows/macOS vs Linux. This is the one failure that is silent locally and explosive in CI — exactly the 2am incident. A names it plainly.

**Weakest assumption.** "This is a default convention for human-created files, not a language-mandated filename." That clean binary — human vs. language-mandated — is false at the edges. Enormous categories are *convention-mandated but not language-mandated*: `docker-compose.yml`, `.github/workflows/*.yml`, CI config, `CODEOWNERS`, Terraform `main.tf`. A's two-bucket model gives no guidance for the file that a *tool* (not a *language*) expects in a particular shape. When it breaks, the engineer doesn't know which bucket they're in.

**Hidden assumptions.**
1. **That readability in paths is a primary decision criterion.** A repeats "scannable" and "readable" as if humans read raw paths often. On-call engineers mostly *grep, tab-complete, and click stack traces* — they rarely eyeball a directory tree aesthetically. Readability is real but ranks below collision-safety and toolchain-fit, and A elevates it.
2. **That "existing conventions win" is self-executing.** A asserts precedent takes precedence but never says *who adjudicates* a repo with mixed history. In practice "consistency should win" becomes "everyone picks the local majority they happen to see," which perpetuates drift.

**Overlooked risks.** A never addresses the **half-migrated repo** — the state where `MyWork/` and `my-work/` coexist because someone applied the rule "prospectively" to a subtree. That's not a hypothetical; it's the *direct consequence* of A's own "apply prospectively" advice, and it manufactures the exact case-collision A warns against. Proposal B caught this; A did not.

---

## Proposal B

**Steelman.** If the environment is confirmed mixed Windows-dev + Linux-CI (it is — the global rules call out PowerShell-default with Bash for CI-style tasks), then B's central insight is the correct organizing principle: **the separator choice is a semantic-channel allocation problem, not a style preference.** `.` = extension boundary, `_` = "a toolchain needs this," `-` = everything else. That reframing is what makes the exceptions *derivable* instead of memorized. For B to be exactly right, the reserved-channel discipline has to hold under pressure — engineers must actually reach for the "is a toolchain forcing my hand?" test before naming.

**Assessment: Agree**, with one correction below. B is the stronger proposal because it converts a taste question into a rule with a decision procedure. "Seeing `test_helpers.py` tells you it's Python; `test-helpers.md` tells you it's prose" — that's the underscore doing *information* work, not decoration. That is exactly the explicit-contract-over-convention property I optimize for.

**Strongest point.** "**Overloading `_` destroys the signal.**" This is the argument A missed entirely. If underscore is both the generic word-separator *and* the Python-module marker, the character carries zero information — you can no longer glance at a name and infer its nature. Reserving `_` makes every underscore *mean something*. That's a maintainability asset that compounds over three years.

**Weakest assumption.** "Files that must be importable as code modules are a **bounded minority**." This is ecosystem-dependent and possibly wrong here. This is a heavily scripting/automation-oriented environment (PowerShell modules, Python tooling, `corp-ops/scripts/`). PowerShell `.psm1` modules, dot-sourced `.ps1` files, and Python packages may be a *larger* share than B assumes. If the importable-file fraction is actually 30–40%, the "one rule covers ~90%" claim collapses toward "you're constantly in exception territory," and the two-tier rule feels less like an edge case and more like a coin-flip on every file.

**Hidden assumptions.**
1. **That the memory-system's kebab-case slug precedent generalizes to filenames.** B leans on "the ecosystem already voted" — but the memory slugs are *logical identifiers in frontmatter*, not filesystem paths subject to import mechanics. Citing them as binding precedent for `.py`/`.psm1` naming is a category error; those slugs never had to be importable. The precedent supports kebab for *docs and configs*, which was never the contested ground.
2. **That "write the rule down, live with it, then automate" will actually reach the automation step.** B defers the pre-commit hook — reasonable — but assumes the catalog-then-enforce sequence completes. In practice deferred automation in a solo/small-team context is deferred *forever*, and B's own "silent drift back to mixed conventions (likelihood: high without automation)" risk then materializes by default. B correctly diagnoses the risk and then prescribes the exact condition that triggers it.

**Overlooked risks.** B is silent on **PowerShell specifically**, which is the *primary shell here*. PowerShell command/parameter parsing and module auto-loading have their own quirks — a hyphen in a filename is fine, but PowerShell *cmdlet naming* is `Verb-Noun` (hyphenated!) while its *module files* often follow the module name. B's "underscore where a toolchain requires it" rule was written with Python in mind and never checks whether the dominant shell's module/dot-sourcing conventions fit the same carve-out. The rule might be right for PowerShell too — but B didn't verify it, and the on-call engineer inherits that unverified gap.

---

## My revised recommendation

**I maintain kebab-case, lowercase, and I adopt Proposal B's framing as the reasoning — the semantic-channel model is what promotes this from convention to contract.** Neither proposal changed the *answer*; Proposal B changed how I'd *justify and teach* it, which for a three-year-maintainability question is the part that matters.

What updated me: **B's "reserved channels" argument.** A gave me a rule and a list of exceptions. B gave me a rule from which the exceptions are *derivable* — `_` is load-bearing, so any file whose *toolchain* assigns meaning to its name is automatically the exception, and everything else is kebab. An on-call engineer at 2am can *reconstruct* that policy from first principles; they cannot reconstruct A's enumerated exception list. Derivable beats memorized.

Where I overrule both:

1. **The exception test must be "toolchain-load-bearing," not "language-mandated."** Both proposals said "language" and both missed CI/infra files (`docker-compose.yml`, workflow YAML, `main.tf`). State it as: *if a tool reads meaning from this file's name, match what the tool expects; otherwise kebab-case, lowercase.* That closes A's human-vs-language gap and B's Python-centric blind spot in one clause.

2. **Verify the PowerShell case before publishing, not after.** This ecosystem is PowerShell-first. Before this becomes canon, confirm the carve-out covers `.psm1`/`.ps1` dot-sourcing and module-manifest naming. I will not ship a naming standard whose reserved-underscore rule was validated only against Python in a shop that runs on PowerShell. That's a 15-minute check that prevents a stated policy from being quietly wrong for the most common file type here.

3. **Do not defer enforcement indefinitely, but do not automate blind.** B's "catalog then hook" is right; A has no enforcement story at all. The concrete commit: catalog exceptions for **two weeks**, then ship the pre-commit hook. Put a date on it, because "deferred until exceptions are known" with no deadline is how B's own high-likelihood drift risk wins.

4. **Reject "apply prospectively" as stated.** A's prospective-only advice manufactures half-migrated trees — the precise case-collision source. Adopt B's rule instead: **per subtree, all-or-nothing.** Either a directory is fully converted or it's left entirely alone. Never leave a folder straddling both conventions, because that straddle *is* the 2am incident.

Net: **B is the better-reasoned proposal and becomes the spine of the standard.** A's only unique contribution — "don't rename established dirs" — is real but must be tightened from "prospective" to "wholesale-per-subtree" or it backfires.


### Openai (gpt-5.6-terra)

## Proposal A

**Steelman first:** This is exactly right if the ecosystem is mixed-language, paths commonly appear in URLs/package names, and the team values a simple prospective default over migration purity. Tool-required exceptions stay narrow and documented.

**Your assessment:** **Agree.** Lowercase kebab-case is the most practical default for human-named, non-importable files and directories. Its key strength is that it works without forcing one language’s conventions onto everything else.

**Strongest point:** “Existing repository conventions take precedence.” This avoids breaking imports, scripts, CI, links, and team muscle memory for a cosmetic gain.

**Weakest assumption:** That Python modules are only an exception. If the ecosystem is mostly Python, importable paths are common enough that `snake_case` may be the operational default, not an edge case. Treating them as exceptional would create constant policy friction.

**Hidden assumptions:**

- The team can reliably distinguish general-purpose files from files that will become importable modules.
- A documented convention will be applied consistently without lightweight automation or review guidance.

**Overlooked risks:**

- Kebab-case can be awkward for tool-generated references and shell variables derived from filenames; those integrations need explicit handling.
- “General-purpose” is underspecified: scripts, infrastructure modules, test helpers, and configuration fragments do not all have the same ecosystem conventions.

## Proposal B

**Steelman first:** This is exactly right if the organization controls a mixed Windows/Linux ecosystem, most new paths are documentation, configuration, automation, or web-facing assets, and it needs a strict rule that prevents case-related CI failures while deferring enforcement until real exceptions are known.

**Your assessment:** **Partially agree.** I agree with lowercase kebab-case as the default and with avoiding mass renames. I disagree with the claim that underscores should be reserved only for toolchain requirements, and with presenting several preferences as hard technical facts.

**Strongest point:** The proposal correctly treats migration restraint as an operational requirement: do not partially rename legacy trees and create a long tail of broken references.

**Weakest assumption:** That underscore has a stable, useful semantic meaning across the ecosystem. It does not. `_` is widely conventional in Python, generated assets, tests, internal files, and tooling—but it does not reliably mean “language-mandated.” Enforcing that distinction creates review debates without operational value.

**Hidden assumptions:**

- The cited “memory subsystem” convention is authoritative precedent for all repositories and artifact types.
- A case-collision policy is primarily solved by the separator. It is actually solved by a lowercase-only rule and repository hygiene; kebab-case itself is not the decisive control.

**Overlooked risks:**

- A blanket rule for scripts may conflict with established command, framework, or package conventions, creating needless exceptions.
- “No quoting required” is overstated: filenames can still require quoting based on context, and names beginning with `-` remain command-option hazards.
- Deferring enforcement indefinitely risks drift; the proposal needs a clear owner, adoption date, and review checkpoint—not merely “a few weeks.”

## Revised recommendation

**Adopt lowercase kebab-case as the canonical default for new, human-created, non-importable filenames and folders.**

Examples: `payment-retry-policy.md`, `billing-service/`, `sync-mywork.ps1`.

Keep existing repository conventions intact. Use `snake_case` wherever the language, framework, tool, or established local convention expects it—especially importable Python modules. Preserve prescribed names exactly: `README.md`, `Dockerfile`, `package.json`, `__init__.py`, generated and vendored paths.

This maintains the shared conclusion of both proposals, but rejects Proposal B’s artificial underscore reservation. The policy should optimize for adoption: one default, explicit exceptions, no mass rename, and a short written rule added to repository contribution guidance. Lowercase—not the hyphen—is the non-negotiable cross-platform safety control.


## Synthesis (by gemini, non-participant)

## Consensus
- **Lowercase kebab-case (`-`)** is the canonical default separator for general-purpose files and directories.
- **Cross-platform safety:** Forcing lowercase is the actual mechanism that prevents case-collision bugs (e.g., `MyWork/` vs `mywork/`) across Windows/macOS local environments and case-sensitive Linux CI environments.
- **Exceptions are necessary:** Language-mandated or tool-required filenames (e.g., `Dockerfile`, `package.json`, importable Python modules) must be allowed to bypass the rule.
- **No blind mass-renaming:** Widespread renaming of established legacy repositories should be avoided to prevent breaking imports, scripts, and muscle memory. 

*(Note: Consensus was driven by strong shared evidence regarding filesystem limitations, CI incident patterns, and web/URL compatibility.)*

## Unresolved Disagreements
**Crux 1: The semantic role of the underscore (`_`)**
- **The Split:** One side argued `_` should be strictly reserved as a "toolchain-required" marker, giving the character a semantic meaning (hyphen = human/prose/generic, underscore = machine-required). The other side argued this is an artificial reservation, as `_` is used loosely across tooling, and enforcing a strict semantic divide creates unnecessary review friction.
- **Assessment:** The semantic reservation argument is stronger. Defining a strict boundary ("only use `_` if a tool expects it") transforms a stylistic preference into a derivable rule. It reduces cognitive load by giving engineers a specific test to apply when naming a file, rather than forcing them to consult a memorized list of exceptions.

**Crux 2: Migration strategy for legacy directories**
- **The Split:** One side advocated for a "prospective-only" application (new files get kebab-case, existing files are ignored). The other side argued for "all-or-nothing per subtree."
- **Assessment:** The "all-or-nothing" argument is substantially stronger. "Prospective-only" changes inside an existing directory manufacture the exact half-migrated states (e.g., `MyWork/` and `my-work/` coexisting) that cause the cross-platform case collisions this policy is designed to eliminate.

## Argument Quality Assessment
- **Best-reasoned:** The argument analyzing the migration strategy and semantic channels was the best reasoned. It correctly identified that lists of exceptions fail in practice without an underlying derivable principle, and it mapped the consequences of partial migrations directly to CI failures.
- **Strongest argument:** The refutation of "apply prospectively." Pointing out that prospectively applying lowercase kebab-case to a CamelCase directory creates a split-case environment that will eventually detonate on a case-sensitive Linux filesystem is airtight reasoning.
- **Weakest argument:** The reliance on "readability" and "scannability" as primary decision criteria. As pointed out in the debate, on-call engineers interact with paths via `grep`, tab-completion, and stack traces—programmatic safety and toolchain compatibility vastly outweigh aesthetic scannability.

## Blind Spots
- **PowerShell Validation:** Despite acknowledging the environment is heavily PowerShell-oriented, neither participant confidently verified how PowerShell's module auto-loading, dot-sourcing, or `Verb-Noun` conventions interact with the proposed naming rules. A standard was nearly finalized without checking the dominant scripting language's edge cases.
- **Leading Hyphens:** While hyphens were debated as *separators*, the hazard of files *starting* with hyphens (which break shell command parsing by mimicking flags) was barely touched upon. 
- **Framework vs. Language vs. Tool:** The debate conflated "language-mandated" (e.g., Python imports) with "tool-mandated" (e.g., `docker-compose.yml`), but largely missed "framework-convention" (e.g., Next.js `[slug].tsx` routing). The exact boundaries of what constitutes a valid "toolchain requirement" remain slightly fuzzy.

## Recommended Decision
Adopt **lowercase kebab-case (`-`)** as the canonical default separator for all new, general-purpose files and directories.

- **The Rule:** If a tool, framework, or language *reads meaning* from the filename or requires a specific format to function properly (e.g., Python modules, `main.tf`, `README.md`), use the exact format the tool expects (including `snake_case` or `camelCase`). For everything else (docs, generic scripts, config fragments, generic folders), use lowercase kebab-case. No filenames may start with a hyphen.
- **Migration:** Apply changes using an "all-or-nothing per subtree" approach. Do not mix conventions within a single directory.
- **Rationale:** Lowercase eliminates Windows/macOS vs. Linux CI collisions. Hyphens are web/URL-safe and standard for generic files. Reserving the underscore specifically for toolchain requirements gives engineers a simple, derivable test for naming files at 2 AM.
- **Objections Addressed:** I acknowledge the objection that reserving `_` is "artificial." However, operational consistency requires clear boundaries. Treating `_` as a load-bearing indicator of a tool's requirements is a highly useful artificial constraint that prevents silent convention drift.

## Risks
- **PowerShell Conflicts:** The policy might clash with native PowerShell auto-discovery or module conventions, breaking existing automation.
- **Enforcement Drift:** If automation is deferred indefinitely, the ecosystem will quietly revert to mixed conventions.
- **Accidental Breaking Changes:** Overzealous renaming of legacy trees to fit the new policy could break deeply buried CI scripts, imports, or documentation links.

## Action Items
1. **Verify PowerShell:** Before publishing the policy, conduct a 15-minute verification on how PowerShell `.psm1`, `.psd1`, and `.ps1` files behave with kebab-case vs snake_case in this specific environment to ensure no unexpected friction.
2. **Publish the Standard:** Document the rule, emphasizing the "toolchain load-bearing" exception test, the lowercase requirement, the subtree-migration rule, and the prohibition on leading hyphens.
3. **Catalog Exceptions:** Announce a two-week grace period to catalog genuine toolchain exceptions (e.g., specific CI tools, framework routing files, language imports).
4. **Automate:** At the exact end of the two weeks, deploy a pre-commit hook that enforces lowercase kebab-case, hardcoding the discovered toolchain exceptions as bypasses. Do not defer this step.
