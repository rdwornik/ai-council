# AI Council Debate: What is the canonical filename / foldername separator across the Dev ecosystem?

**Panel:** claude-opus-4-8, gpt-5.6-terra
**Synthesizer:** gemini (non-participant)
**Rounds:** 2
**Panel Mode:** custom
**Debate Mode:** pick
**Source:** cli

## Verdict Summary

**Decision:** Implement a two-part standard that explicitly separates and answers both interpretations of the prompt:
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

Use the forward slash (`/`) as the canonical path separator in all code, config, documentation, and internal tooling. Treat the backslash as a platform-specific artifact that gets normalized at the boundary, never propagated inward.

## Alternatives Considered

- **Backslash (`\`) as canonical, normalize to forward slash on non-Windows:** Rejected. This inverts the burden — you'd be normalizing on the majority of platforms (Linux CI runners, containers, macOS dev machines) and it fights every URL, POSIX API, Git internal, and web standard that already assumes `/`. You'd be swimming upstream against the entire toolchain.

- **"It depends — let each layer use its native separator":** Rejected. This is the status quo that produces the 2am pages. Path strings cross module boundaries, get logged, get stored in databases, get embedded in JSON, and get compared as strings. If separators aren't canonical, you get non-reproducible bugs where a path matches on one machine and not another. Ambiguity at boundaries is exactly the cognitive-load debt I'm trying to avoid.

## Decision Criteria

- **Cross-platform reproducibility:** the same path string must behave identically on Linux, macOS, and Windows dev/CI environments.
- **Compatibility with adjacent standards:** URLs, URIs, Git, Docker, and virtually all package managers use `/`. The canonical choice should not require translation to interoperate with these.
- **2am debuggability:** a path in a log line should be unambiguous and copy-pasteable into a shell without escaping surprises.
- **Migration cost:** the choice should require normalizing at the fewest, most controllable points.

## Assumptions

- Your ecosystem runs primarily on POSIX systems (Linux containers, CI runners) with Windows present mainly on developer workstations. If Windows is your *production* target, the normalization boundary shifts but the canonical choice does not.
- Path strings are exchanged between components (logs, configs, serialized data), not just used ephemerally within a single OS API call.
- You control your own code and can enforce a boundary policy; you're not fully at the mercy of a third-party library that emits backslashes internally.

## Recommendation

**Do now:**
- Declare `/` canonical in your engineering standards. All literals, config keys, glob patterns, and stored path data use `/`.
- Normalize at ingress: any path arriving from a Windows filesystem API, `os.getcwd()`, drag-and-drop, or user input gets converted to `/` immediately (e.g., `.replace('\\', '/')` or your language's `PurePosixPath` / `path.posix` equivalent).
- Convert back to native only at egress — the exact moment you hand a path to a native OS call that requires it. On most platforms this conversion is a no-op.
- Add a lint rule or CI check flagging backslash literals in path-typed code.

**Do not do now:**
- Don't sprinkle `os.path.sep` / `path.sep` everywhere "for portability." That's the trap: it makes the *runtime* portable but makes the *stored strings* non-portable, which is worse.
- Don't build a custom Path abstraction layer. Use the standard library's path type (Python `pathlib`, Node `path.posix`, Go `path` vs `filepath`).

**Defer:**
- Rewriting legacy modules that already work with mixed separators — only migrate them when touched, guarded by the lint rule.

## Rationale

- **The entire adjacent ecosystem already voted for `/`.** URLs, Git object paths, Docker `COPY` directives, YAML/JSON config, S3 keys, and glob syntax all use `/`. Choosing anything else means you translate at every one of those boundaries — dozens of touchpoints vs. one (the native-OS-call egress).
- **Windows accepts `/` natively in nearly all its APIs.** `CreateFile`, .NET, and PowerShell handle forward slashes. The backslash is a display convention, not a hard requirement — which means "canonical `/`" costs almost nothing even on Windows.
- **Backslash is an escape character in most string contexts.** A path in a JSON log line, a regex, or a shell command with `\` invites double-escaping bugs. `/` never needs escaping. This directly reduces 2am confusion.
- **Normalize-at-boundary is a single mental model.** An engineer joining in year three needs to learn one rule: "internal is always `/`, convert only when calling the OS." That's a far lower cognitive load than "check the platform everywhere."
- **String comparison actually works.** Deduplicating, caching, and matching paths by string equality only works if separators are canonical — otherwise `a/b` and `a\b` are two cache entries for one file.

## Trade-offs

- You give up "native-looking" paths in Windows-facing logs and error messages; a Windows user may see `C:/Users/foo` instead of `C:\Users\foo`. Cosmetic, and Windows tools accept it.
- You take on a small, explicit normalization step at every ingress point (~1 line per boundary). This is deliberate — you're concentrating the complexity where it's visible rather than diffusing it.
- Raw UNC paths (`\\server\share`) and Windows extended-length prefixes (`\\?\`) are the genuine exceptions and must be handled explicitly at the OS boundary, not normalized blindly.

## Risks

- **A third-party library emits backslashes internally and you compare its output as strings** — *likelihood: medium, impact: medium.* Mitigate by normalizing at the library boundary, same as any other ingress.
- **Blind `\` → `/` replacement corrupts UNC/extended-length paths** — *likelihood: low, impact: high.* Mitigate with a normalization helper that detects and preserves these prefixes, rather than a raw string replace scattered in code.
- **Team drifts back to `os.path.sep` habits without enforcement** — *likelihood: high, impact: low-medium.* Mitigate with the CI lint rule; this is the single most important durable control.

## Signals to Revisit

- Windows becomes your **primary production runtime** and you're paying measurable overhead or hitting API edge cases from forward-slash paths.
- A core dependency in your stack hard-fails on `/` (rare, but would force a boundary redesign).
- You start dealing heavily with UNC/network paths where the `\\` prefix is semantically load-bearing — at that point you may need a richer path type than plain strings, not a different separator.


### Openai (gpt-5.6-terra)

## Position
Use kebab-case (`kebab-case`) as the canonical separator for new folders and general-purpose filenames. Preserve ecosystem-mandated conventions where tooling or frameworks require them, rather than forcing one rule everywhere.

## Alternatives Considered
1. **snake_case (`snake_case`) everywhere** — Rejected because it conflicts with dominant JavaScript/TypeScript package, URL, Docker, and modern web-project conventions; it also looks inconsistent beside common frontend tooling.
2. **camelCase (`camelCase`) everywhere** — Rejected because it is less shell-friendly, harder to scan in long directory names, and clashes with common conventions for package names, URLs, and non-code assets.

## Decision Criteria
- Consistency across a mixed modern development ecosystem, especially JavaScript/TypeScript, web, containers, CI, and documentation.
- Compatibility with package registries, URLs, command-line tooling, and case-sensitive filesystems.
- Low cognitive and enforcement overhead for a typical engineering team.
- Respect for framework- and language-specific conventions that are operationally significant.
- Avoiding cross-platform filename problems, particularly case-only differences on macOS and Windows.

## Assumptions
- The organization has a heterogeneous codebase or expects to support multiple languages and frameworks.
- This decision covers repository folders and ordinary filenames, not necessarily source-code symbol names.
- The team can document and lint a small naming policy but should not build custom enforcement tooling.
- Existing repositories do not require a disruptive mass rename.

## Recommendation
**What to do now**
- Adopt **lowercase kebab-case** for new repository directories and general filenames: `payment-reconciliation/`, `release-notes.md`, `user-profile.json`.
- Use only lowercase ASCII letters, digits, and hyphens for portable paths where practical.
- Add the rule to the engineering conventions document and enforce it in code review.
- Use an explicit exception rule: follow the naming convention required by the relevant ecosystem or framework.

Examples of valid exceptions:
- `package.json`, `tsconfig.json`, `Dockerfile`, `Makefile`, `.github/`
- Python modules/packages: `snake_case.py`, `snake_case/`
- Go source files: `snake_case.go`; Go package directory names commonly remain short lowercase names.
- Java classes: `PascalCase.java`
- Framework-reserved names such as Next.js `page.tsx`, `layout.tsx`, or Rails conventions.

**What not to do now**
- Do not rename existing files solely to achieve visual consistency.
- Do not impose kebab-case on language source files or framework-reserved paths.
- Do not create a custom cross-language filename linter unless naming defects are demonstrably causing build or deployment incidents.

**What to defer until later**
- Introduce automated path-name validation only if repository scale, generated artifacts, or multi-team contribution volume makes review-based enforcement insufficient.
- Define narrower conventions by repository type if a monorepo develops distinct backend, frontend, mobile, and infrastructure ownership models.

## Rationale
- Kebab-case is accepted by npm package naming rules and maps cleanly to URLs, CLI arguments, static-site paths, container-related assets, and documentation filenames.
- Hyphens are more readable than camelCase for multiword paths: `customer-account-migration/` is easier to parse than `customerAccountMigration/`.
- Lowercase kebab-case avoids case-only path collisions such as `UserProfile/` versus `userprofile/`, which can work on Linux but fail or behave unpredictably on default macOS/Windows filesystems.
- A universal rule has diminishing returns when it fights language conventions; forcing Python from `snake_case.py` to `kebab-case.py` creates import friction and requires non-idiomatic handling.
- Documentation plus review is usually enough: naming decisions occur infrequently relative to code changes, so custom tooling is unlikely to justify its maintenance cost for a normal-sized team.

## Trade-offs
- The repository will not be visually uniform: Python, Java, framework-reserved files, and generated files will retain ecosystem-specific naming.
- Developers must understand one simple default plus exceptions rather than applying a rigid universal rule.
- Some tools or teams may prefer underscores for machine-generated files; those should remain exceptions where changing them affects tool compatibility.
- Deferring bulk renames leaves legacy inconsistency in place, but avoids broken imports, stale links, Git history noise, and potentially large merge conflicts.

## Risks
- **Exception sprawl makes the policy ambiguous** — Likelihood: medium; impact: medium. Mitigate with a short table of approved exceptions and require any new exception to cite a tool, language, or framework requirement.
- **Teams apply kebab-case to language files where it is non-idiomatic** — Likelihood: medium; impact: low-to-medium. Mitigate by explicitly listing Python, Java, Go, and framework routing conventions in the policy.
- **Existing inconsistent paths continue causing confusion** — Likelihood: medium; impact: low. Mitigate by applying the standard to new paths and opportunistically correcting a path only when it is already being substantially reworked.
- **Case-sensitive CI or Linux deployment exposes legacy case issues** — Likelihood: low-to-medium; impact: high for affected builds. Mitigate with CI checks for case-collision paths and test builds in Linux containers.

## Signals to Revisit
- A primary ecosystem changes its official convention or imposes a materially different path format.
- More than a few path-related incidents per quarter occur, such as broken imports, case-sensitivity failures, invalid package publication, or deployment routing errors.
- The organization adopts a large monorepo with enough contributors that naming review becomes inconsistent; at that point, add lightweight automated validation.
- A repository becomes predominantly language-specific, such as Python-only or Java-only, where adopting that ecosystem’s native file conventions may provide more value than a cross-ecosystem default.


## Round 2: Critique

### Claude (claude-opus-4-8)

# The Framing Problem

Before assessing either proposal, I have to name the elephant: **these two proposals answer two entirely different questions.** The prompt asks "what is the canonical filename/foldername *separator*." That word is ambiguous, and each proposal resolved the ambiguity differently:

- **Proposal A** read "separator" as the **path separator** — the character *between* directory components (`/` vs `\`).
- **Proposal B** read "separator" as the **word separator** — the character *within* a name (`kebab-case` vs `snake_case` vs `camelCase`).

This is not a minor discrepancy. It means the council is not actually deliberating on one decision. My job here is to assess each on its own terms, but the on-call engineer at 2am doesn't care which interpretation was "intended" — they care whether the resulting standard prevents pages. So I'll evaluate each against that bar.

---

# Proposal A — Forward slash as canonical path separator

**Steelman first:**
This is exactly right if path *strings* cross boundaries — logs, config, serialized data, cache keys, database columns — and get compared, stored, or copy-pasted rather than staying ephemeral inside a single OS call. In that world, canonicalizing on `/` collapses an entire class of non-reproducible "matches-on-my-machine" bugs into a single, enforceable ingress rule. The entire adjacent ecosystem (URLs, Git, Docker, S3, globs) already voted `/`, and Windows accepts `/` in nearly all its APIs, so the choice costs almost nothing where it appears to cost the most.

**Your assessment: Agree.**
This is the correct answer to the question it actually answers. It satisfies my primary lens directly: the year-three engineer learns *one rule* — "internal is always `/`, convert only at the OS boundary." That is genuinely low cognitive load. The "string comparison actually works" argument is the load-bearing one, and it's correct: canonicalization is what makes dedup, caching, and equality-matching sound. Non-negotiable in any system that treats paths as data.

**Strongest point:**
"String comparison actually works." `a/b` and `a\b` being two cache entries for one file is precisely the kind of silent, non-reproducible corruption that produces 2am pages with no stack trace. Canonicalization is the only thing that makes path equality a real operation.

**Weakest assumption:**
"Windows is present mainly on developer workstations, not production." If Windows is the *production* runtime, the normalization boundary moves to a hotter, higher-volume place, and the UNC/extended-length-prefix exceptions (`\\server\share`, `\\?\`) stop being edge cases and become routine. The proposal acknowledges this in "Signals to Revisit," which is honest — but a team that adopts the headline "`/` is canonical" and skips the fine print will blind-replace `\`→`/` and corrupt UNC paths. That failure is low-likelihood/high-impact, exactly the profile that bites at 2am.

**Hidden assumptions:**
1. **Paths are represented as strings, not typed objects.** The whole proposal presumes a world where paths get stringified early and passed around. In a codebase that uses `pathlib.Path` / typed path objects end-to-end, the "canonical separator" question partly dissolves — the type handles it — and the ingress-normalization discipline matters far less. The author never notices they've assumed stringly-typed paths as the baseline.
2. **The team can actually locate every ingress point.** "Normalize at ingress" assumes ingress points are enumerable and controlled. In a large system with many third-party libraries emitting paths, "ingress" is not a clean perimeter — it's a fuzzy, ever-expanding surface. The single-mental-model claim quietly depends on a boundary that may not be crisp in practice.

**Overlooked risks:**
- **Logging normalization changes forensic fidelity.** If you normalize `C:\Users\foo` to `C:/Users/foo` in logs, and an incident involves a genuinely native-path-dependent bug, the log no longer shows what the OS actually saw. Minor, but it's a debuggability cost the proposal frames as purely cosmetic.
- **The lint rule flags literals but not runtime-constructed backslash paths.** A CI check on `\` literals catches source, not paths assembled from library output. The "single most important durable control" has a coverage gap the proposal doesn't flag.

---

# Proposal B — kebab-case as canonical word separator

**Steelman first:**
This is exactly right if the actual pain the org faces is **case-collision and word-separation inconsistency in repo/asset naming** — `UserProfile/` vs `userprofile/` breaking between Linux CI and macOS/Windows dev, or ad-hoc mixing of camelCase and snake_case in URLs and static assets. In a heterogeneous JS/TS-heavy ecosystem where filenames become URLs and package names, lowercase-kebab plus a small documented exception table is a genuinely low-overhead, review-enforceable standard that respects the languages that have non-negotiable conventions.

**Your assessment: Partially agree.**
The content is sound *for the question it answers*, but I disagree that it answers the question asked. "Separator" in "filename/foldername separator" most naturally means the path separator — the thing that separates one name from the next — not the intra-name word delimiter. Proposal B is a well-constructed answer to a question that wasn't cleanly posed. On its own terms it's competent; as a response to the council question it's a category substitution the author never acknowledges.

**Strongest point:**
The case-collision argument. Lowercase-kebab structurally eliminates `UserProfile/` vs `userprofile/` — a real, high-impact/low-likelihood failure that passes on Linux and detonates on default macOS/Windows filesystems. That's the one part of this proposal that directly prevents a 2am page.

**Weakest assumption:**
"Documentation plus review is usually enough; naming decisions occur infrequently." This is the assumption most likely to be false and it's the one my entire lens rejects. Review-based enforcement of a convention-with-exceptions decays. The exception table grows, reviewers rotate, and by year three nobody agrees on whether a new file qualifies for an exception. Naming decisions are *not* infrequent in a growing monorepo — every new file is one. If this assumption is false, you don't get a hard failure; you get slow, ambiguous drift, which is worse because there's no incident to trigger a fix.

**Hidden assumptions:**
1. **Filenames are frequently also URLs and package names.** The kebab preference is *justified* by npm/URL/static-site friendliness, which silently assumes a web-facing ecosystem where the filename escapes into a URL. For a backend-only or data-pipeline org, that justification largely evaporates and snake_case would be equally defensible.
2. **A "small" exception table stays small.** The author's own top risk is "exception sprawl," yet the recommendation structurally *depends* on exceptions staying enumerable. That's an unexamined bet against the proposal's own identified risk.

**Overlooked risks:**
- **The default-plus-exceptions model has no crisp decision procedure.** "Follow the ecosystem convention" is not a rule an engineer can mechanically apply at 2am — or at code-review time. When is `my-config.json` correct vs `myConfig.json`? The policy offloads a judgment call to the reviewer, which is exactly the diffuse cognitive load I treat as debt.
- **Silence on generated artifacts.** Build tools emit files with their own naming (hashed bundles, `snake_case` codegen). The proposal mentions generated files only in passing and never defines whether the standard governs them, guaranteeing recurring "does this count?" debates.

---

# My Revised Recommendation

**I maintain that the correct canonical *separator* — the direct answer to the question — is Proposal A's forward slash (`/`), and I did not update off it.**

Neither proposal changed my mind on the path-separator question, because Proposal B never contested it — it answered a different question. And Proposal A's core argument (canonicalize so string equality is sound; normalize at one visible boundary; ride the ecosystem that already chose `/`) is precisely the "concentrate complexity where it's visible" discipline my lens demands. Nothing in either document undermines it.

**What the two proposals together forced me to sharpen — this is my update:**

The question conflates two orthogonal decisions, and a serious engineering standard must resolve **both** explicitly, because both produce 2am pages by different mechanisms:

1. **Path separator (between components): `/` is canonical.** Adopt Proposal A wholesale, *including its fine print* — the UNC/`\\?\` exceptions are not optional footnotes; they are the part that will corrupt data if ignored. The lint rule is the durable control. Non-negotiable addition: the standard must state whether paths are strings or typed objects, because Proposal A's entire model assumes stringly-typed paths and that assumption should be *chosen*, not defaulted into.

2. **Word separator (within a name): lowercase-kebab as default is acceptable, but Proposal B's enforcement model is not.** The specific evidence that moved me is Proposal B's own listed top risk — "exception sprawl" — sitting directly beside its recommendation to enforce via review alone. That's an internal contradiction. If exception sprawl is the named failure mode, then "documentation plus review is enough" cannot also be true. **The word-separator rule must be machine-enforced from day one** (a filename linter and a case-collision CI check), not "deferred until incidents occur." A convention-with-exceptions that relies on human vigilance is a convention that has already decided to rot.

**Where I break with Proposal B without softening it:** its "defer automated validation" recommendation is wrong on my lens. The case-collision check specifically is cheap, prevents a high-impact cross-platform failure, and has no reason to wait for an incident. Deferring it trades a one-time CI setup cost against a class of 2am pages. That trade is backwards.

**Bottom line:** Adopt Proposal A as the separator answer. Adopt Proposal B's *content* (lowercase-kebab, ecosystem exceptions) but reject its *enforcement posture* — lint and CI-check both rules now, because the year-three engineer inheriting a review-enforced exception table inherits ambiguity, and ambiguity is the debt.


### Openai (gpt-5.6-terra)

## Proposal A

### Steelman first

Proposal A is exactly right **if the question is actually about directory-path separators**—for example, whether internal serialized paths should be `src/components/button.tsx` or `src\components\button.tsx`.

Under that interpretation, a cross-platform ecosystem primarily running in Linux containers, CI, Git, URLs, package tooling, and cloud services should canonicalize exchanged path strings to `/`, normalize external/native input at carefully implemented boundaries, and avoid allowing platform-specific separators into stored state, cache keys, manifests, or protocols.

### Your assessment

**Disagree with its Position as an answer to this question.**

It answers a different question.

“Filename / foldername separator” conventionally means the separator between words in a file or directory name: `user-profile`, `user_profile`, or `userProfile`. Proposal A instead addresses the **path-component separator**: `/` versus `\`.

Its technical conclusion on path serialization is broadly sound, but adopting it as the council’s answer would leave the actual naming decision unresolved. A team following it still does not know whether to create `payment-reconciliation/`, `payment_reconciliation/`, or `paymentReconciliation/`.

### Strongest point

The distinction between **canonical exchanged/stored path representations** and **native filesystem API representations** is valuable. For paths that cross process, machine, repository, database, URL, or config boundaries, `/` is the right canonical separator in almost every modern development ecosystem.

### Weakest assumption

The weakest assumption is that every incoming filesystem-looking string can safely be normalized into a slash-delimited internal path.

What breaks: Windows paths are not merely POSIX paths with different separators. Drive-relative paths (`C:foo`), UNC paths (`\\server\share`), extended-length paths (`\\?\...`), device paths, reserved names, and case behavior require parsing under Windows semantics. A simple normalization helper is insufficient if the team genuinely supports these forms. Worse, the proposal itself acknowledges this risk but still operationally recommends a model that encourages ordinary strings as a path interchange format.

### Hidden assumptions

1. **That path strings should be compared for identity.**  
   Separator normalization alone does not make string equality safe. Relative versus absolute paths, `.` and `..`, symlinks, drive-letter casing, Unicode normalization, filesystem case sensitivity, and trailing separators can all make two strings refer to the same object—or one string resolve differently in different contexts.

2. **That Windows accepts `/` consistently enough to make egress conversion negligible.**  
   Many Win32 APIs accept forward slashes, but not all consumers do. Command-line tools, legacy tools, argument parsers, shell contexts, and third-party libraries can interpret `/` as an option prefix or otherwise mishandle it.

### Overlooked risks

- **Conflating filesystem paths with URLs, Git paths, glob syntax, and object-store keys.** They all use `/`, but they have different parsing and normalization rules. A filesystem path must not be normalized as though it were a URL or S3 key.
- **Security-sensitive path handling.** Normalizing separators does not address traversal (`..`), symlink escapes, alternate data streams, reserved Windows device names, or canonicalization-before-authorization bugs.
- **Operational overreach.** A lint rule banning backslash literals will create false positives in regexes, escaping, Windows-specific test fixtures, documentation, and legitimate OS-boundary code unless path typing and lint configuration are mature.
- **Unnecessary abstraction for a normal team.** This can be a good platform rule, but not necessarily a first-order engineering-standard investment unless the organization actually serializes and compares filesystem paths across platforms.

---

## Proposal B

### Steelman first

Proposal B is exactly right if the organization needs a practical, cross-language default for **new ordinary repository folders and non-language-specific filenames**, while preserving conventions that are technically required or strongly idiomatic in individual ecosystems.

In a heterogeneous ecosystem—JavaScript/TypeScript, infrastructure, docs, containers, Python, Go, Java, framework routing, generated assets—the most workable rule is:

- Default new general-purpose directories and filenames to lowercase kebab-case.
- Use ecosystem-required or idiomatic exceptions where a language, framework, tool, or generated artifact dictates another form.
- Do not mass-rename existing paths.
- Avoid custom enforcement until review demonstrably fails at the organization’s scale.

That is a rule a normal team can understand and operate without spending more effort on governance than the inconsistency costs.

### Your assessment

**Agree with its Position.**

This proposal answers the question asked and does so with appropriate operational pragmatism. It recognizes that filename conventions are constrained by imports, framework routing, package managers, generated files, and language norms. A rigid “one separator everywhere” policy would create compatibility work and exceptions large enough to defeat the point of the policy.

Lowercase kebab-case is the best default for portable, human-facing, general-purpose paths. It is readable, URL-friendly, shell-friendly, and avoids case-only collisions. More importantly, Proposal B correctly refuses to force that convention onto source files and reserved paths where the ecosystem already has a stronger convention.

### Strongest point

The strongest point is that **a default with explicit ecosystem exceptions is more operationally consistent than a universal rule that fights the tools**.

Forcing Python modules to kebab-case, Java class files to lowercase kebab-case, or framework-reserved routes into a house style creates avoidable import, routing, build, and onboarding friction. “Uniform appearance” is not a meaningful win if it makes ordinary tool usage non-idiomatic.

### Weakest assumption

The weakest assumption is that documentation and code review will be sufficient enforcement for a “typical engineering team.”

What breaks: as repositories, contributors, automation, and generated content grow, reviewers will miss naming drift. The eventual result is not a controlled exception system; it is inconsistent names with no reliable distinction between “required exception,” “legacy path,” and “someone’s preference.” This is especially likely in monorepos and repositories accepting contributions from multiple teams.

That said, Proposal B appropriately defers automated enforcement rather than prematurely building it. The policy should be written so lightweight validation can be added later without redefining the rule.

### Hidden assumptions

1. **That kebab-case is valid for the relevant non-code paths.**  
   It is usually valid, but some tools encode semantics in underscores, dots, casing, or fixed names. The exception policy needs to cover not just framework-required names, but tool-generated and externally consumed filenames.

2. **That “general-purpose filename” is obvious to contributors.**  
   It often is not. Is `payment-reconciliation.sql` a general artifact, a migration governed by a migration framework, or a database convention? Is `user-profile.proto` a language source file or a schema artifact? Without a short decision rule, engineers will classify similar files differently.

### Overlooked risks

- **Hyphen ambiguity in CLI and scripting contexts.** A file beginning with `-` is hazardous; this proposal should explicitly prohibit leading hyphens even though normal kebab-case does not imply them.
- **Portability beyond case sensitivity.** Lowercase ASCII and hyphens are good defaults, but portable naming also benefits from avoiding spaces, control characters, trailing dots/spaces, Windows-reserved names (`CON`, `NUL`, etc.), and punctuation with shell or glob meaning.
- **External interface stability.** Filenames can be APIs: deployment references, URLs, import paths, scripts, documentation links, CI rules, artifact consumers, and customer-facing download links may depend on them. “Rename when substantially reworked” still needs compatibility awareness.
- **Exceptions need a decision owner.** “Follow the ecosystem” can become a loophole unless exceptions must be demonstrably required or documented as an established language/framework convention.
- **Generated files.** They should generally follow their generator’s convention and should not be hand-renamed. This deserves explicit treatment.

---

# Revised recommendation

**Adopt lowercase kebab-case as the canonical word separator for new general-purpose filenames and folder names: `payment-reconciliation/`, `release-notes.md`, `user-profile.json`.**

This is a **maintained position**, not a compromise between the proposals. Proposal B reinforces it. Proposal A did not change it because Proposal A addresses a separate path-serialization question, not the requested filename/foldername word-separator convention.

## Policy

### Default

For newly created, general-purpose repository paths:

- Use lowercase ASCII letters, digits, and hyphens.
- Use kebab-case for multiword names.
- Do not use spaces, underscores, camelCase, or PascalCase as the general default.
- Do not start names with `-`.
- Avoid case-only distinctions.

Examples:

```text
payment-reconciliation/
release-notes.md
customer-export.json
build-scripts/
```

### Required exceptions

Follow language, framework, tool, package, generated-file, or externally defined conventions where they apply:

```text
package.json
tsconfig.json
Dockerfile
Makefile
.github/
snake_case.py
PascalCase.java
page.tsx
layout.tsx
```

The exception test is not “I prefer it.” It is: **Would using kebab-case break tooling, violate a recognized ecosystem convention, or make ordinary development materially non-idiomatic?** If yes, use the ecosystem convention.

### Existing paths

Do not mass-rename legacy files for cosmetic consistency. Rename only when a path is already being materially reorganized and all references, imports, URLs, scripts, and consumers can be updated safely.

### Enforcement

Start with a one-page convention and review enforcement. Do not build custom linting now.

Add lightweight automated validation only when evidence shows review is not sufficient—for example, recurring naming-related build failures, case-collision incidents, or a monorepo/contributor scale where drift is measurable.

## Separate but related rule

For serialized, documented, URL-like, Git-like, or cross-platform **path strings**, use `/` as the canonical path-component separator. That is a separate standard from filename word separation. Do not confuse:

```text
payment-reconciliation/config-file.json
```

- `/` separates path components.
- `-` separates words within a filename or folder name.

That distinction should be explicit in the engineering standards to prevent Proposal A’s technically valid concern from being mistaken for the naming decision.


## Synthesis (by gemini, non-participant)

## Consensus
Both participants independently recognized that the prompt's use of the word "separator" was ambiguous and conflated two entirely different engineering concepts:
1. **The Path Separator:** The character used to delimit directories in a file path (`/` vs `\`).
2. **The Word Separator:** The character used to delimit words within a single file or directory name (`kebab-case` vs `snake_case`).

Once this framing issue was identified, both sides reached a strong, evidence-driven consensus on the technical conventions for both:
- **For Path Separators:** The forward slash (`/`) is the canonical standard. Exchanged path strings (in logs, databases, JSON, cache keys) must use `/`. Backslashes are treated as a Windows-specific native artifact, handled only at the OS boundary.
- **For Word Separators:** Lowercase `kebab-case` (`-`) is the optimal default for general-purpose files and folders. However, strict ecosystem-specific exceptions (e.g., `snake_case.py`, `PascalCase.java`, `Dockerfile`, `page.tsx`) are mandatory, as forcing a universal house style would break tooling and framework routing.
- **Legacy Code:** Do not mass-rename existing files just for cosmetic consistency, as this causes merge conflicts and breaks historical links. 

## Unresolved Disagreements
**Crux: The enforcement strategy for the word-separator convention.**
- **OpenAI** argued for deferring automated enforcement. It recommended relying on engineering documentation and human code-review first, introducing custom linting only if scaling issues or naming-related incidents occur. It argued that building a linter for a nuanced, exception-heavy naming rule is an unnecessary operational burden.
- **Claude** argued for immediate machine enforcement (filename linting and case-collision CI checks). It argued that relying on human review for an exception table decays over time, resulting in silent naming drift, ambiguity debt, and cross-platform case-collision bugs.

**Assessment:** Claude's argument is stronger. A convention governed by an evolving list of exceptions is nearly impossible to enforce consistently via human code review. Furthermore, cross-platform case collisions (e.g., a file named `UserProfile` failing on a case-sensitive Linux CI after being authored on case-insensitive macOS) are high-impact, low-visibility bugs that are trivially prevented by a basic CI check. Deferring automation trades a small setup cost for guaranteed future technical debt.

## Argument Quality Assessment
- **Best-reasoned:** Claude in Round 2. Claude brilliantly diagnosed the semantic ambiguity in the prompt, steelmanned the opposing view, and synthesized the two distinct technical problems under a strict, pragmatic lens ("preventing 2am pages"). 
- **Single strongest argument:** Claude's point that *"string comparison actually works"* when path separators are canonicalized to `/`. Caching, deduplication, and equality-matching are fundamentally broken if `a/b` and `a\b` are allowed to coexist in a system's state. Canonicalization is a prerequisite for system stability, not just a cosmetic preference.
- **Single weakest argument:** OpenAI's assumption that "documentation plus review is usually enough" to enforce naming conventions. History in software engineering shows that review-enforced style guides without automated linting inevitably rot, especially when the rule involves a matrix of language-specific exceptions.

## Blind Spots
- **How to mechanize the exceptions:** Neither participant detailed exactly *how* a team should configure a linter to handle the word-separator exceptions (e.g., using a tool like `ls-lint` to map `*.py` to `snake_case` and `*` to `kebab-case`). 
- **Typed Path Objects:** While Claude touched on this, the debate largely assumed paths are handled as raw strings. Standardizing on typed path objects (like Python's `pathlib` or Rust's `PathBuf`) natively solves much of the path separator problem without requiring developers to write manual `.replace('\\', '/')` logic.
- **URL Encoding:** The debate missed that `kebab-case` is not just "URL friendly," but structurally avoids the ugly `%20` or `+` encoding required by spaces, and the visual ambiguity of underscores in hyperlinked text.

## Recommended Decision
Implement a two-part standard that explicitly separates and answers both interpretations of the prompt:

**1. The Path Separator Rule: `/` is canonical.**
All paths stored in state, logs, databases, cache keys, or transmitted across network boundaries must use the forward slash (`/`). Windows backslashes (`\`) are treated strictly as an egress/ingress requirement for native Windows OS APIs. 

**2. The Word Separator Rule: `kebab-case` by default, with rigid ecosystem exceptions.**
All newly created, general-purpose files and folders (e.g., `user-profile.json`, `api-docs/`) must use lowercase ASCII `kebab-case`. You must deviate from this *only* when a specific language, framework, or generator requires it (e.g., `snake_case` for Python, `camelCase` for Go, `PascalCase` for Java classes). 

**3. Enforcement must be automated.**
Do not rely on human memory and PR reviews to enforce naming conventions. Implement CI checks for case-collisions and basic filename linting on day one.

## Risks
- **Data Corruption via Blind String Replacement:** If engineers implement the Path Separator rule by blindly running `.replace('\\', '/')` on all inputs, they will corrupt Windows UNC paths (`\\server\share`) and extended-length prefixes (`\\?\`). 
  - *Mitigation:* Require the use of standard library path-parsing utilities (e.g., Node's `path.posix`, Python's `pathlib`) rather than raw string manipulation.
- **Linter Friction:** A draconian filename linter will block legitimate PRs if an unpredicted exception arises.
  - *Mitigation:* Provide a clear, documented escape hatch (like an `.ignore` file) so developers can unblock themselves while the exception is reviewed.

## Action Items
1. **Update Engineering Standards:** Publish a brief naming and path-handling document that explicitly defines the two distinct rules (Path vs. Word separators) and lists the approved ecosystem exceptions for your stack.
2. **Add a Case-Collision CI Check:** Add a lightweight script to your CI pipeline that fails the build if two paths in the repository differ only by case.
3. **Add a Filename Linter:** Implement a tool like `ls-lint`. Configure the default to `kebab-case` and map your specific ecosystem exceptions (e.g., `*.py` -> `snake_case`).
4. **Configure Linter for New Files Only:** To avoid massive merge conflicts, configure the filename linter to only run against files added or renamed in the current diff. Grandfather in the legacy files.
