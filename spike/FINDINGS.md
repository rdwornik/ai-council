# SPIKE — markdown-it-py vs the hand-written options scanner

**Date:** 2026-07-20 · **Worktree:** `spike-md-parser` · **Status:** THROWAWAY — nothing merges
**Governance:** #77 (merged extractor contract, `94421c2`), #80, #81
**Baseline commit:** `94421c2` · `src/ai_council/output.py` NOT modified · scanner NOT deleted

## Recommendation

**KEEP-SCANNER**, on two independent grounds:

1. It **reintroduces the performance fork the scanner needed 6 terra passes to close** — 30.8×
   slower at 30k chars against a `<1.0s` bound, inside `md.parse()`, unpatchable from our layer.
2. It **does not actually dissolve #81**. It fixes fabrication but triggers the exact inversion
   #81's filing text predicts: a model that fences its whole options list yields `[]` — total
   option loss. #81's done-when requires a fenced list *not* be silently emptied.

It **does** dissolve #80's *rule-authority* question, and the fence-skipping structure is still
worth porting — but the library as a drop-in trades one #81 failure for the other.

## Installed version

| Item | Value |
|---|---|
| `markdown-it-py` | **4.0.0** — **already present**, transitive dep of `rich` (`Required-by: rich`) |
| Install action | **none taken** — no venv install, no `pyproject`/lock edit |
| Ported from | `markdown-it` (JS) **14.1.0**, commit `0fe7ccb`, 2024-03-19 (`markdown_it/port.yaml`) |
| CommonMark spec | highest in-source reference **0.31.2** |
| Preset used | `MarkdownIt("commonmark")` — not the default `gfm-like` (which adds linkify/tables) |

Adoption would carry **zero new dependency cost**: it is already installed and imported.

## Suite parity — NO

Existing `tests/test_output.py`, run unmodified, with `_top_level_bullets` monkeypatched
(`spike/plugin_swap.py`). `output.py` untouched.

| Run | `-k options` | full file |
|---|---|---|
| scanner (baseline) | **46 / 46 pass** | **120 / 120 pass** |
| markdown-it-py | **44 / 46 pass** | **118 / 120 pass** |

Both failures are the same two tests.

### Failure 1 — `test_options_unpaired_and_unequal_delimiters_are_left_alone`

`**uneven*` → scanner `**uneven*`, library `*uneven`.

**The library is spec-correct and the test encodes a bespoke choice.** Reference render is
`<p>*<em>uneven</em></p>`. The scanner's "only equal-length runs pair, anything unpaired is
emitted verbatim" is a deliberate #77 conservatism (an honest `**` beats a silently removed
payload character), not CommonMark. This is a **contract divergence to rule on**, not a
library defect.

### Failure 2 — `test_options_emphasis_unwrapping_is_linear_on_pathological_input`

`"- " + " *a" * 30_000` — bound `< 1.0s`. Scanner **0.12s**, library **3.79s**. See below.

## #81 verdict — **NOT dissolved. Fabrication fixed, but the inversion #81 predicted is REAL.**

> **Correction.** This section first read "library PASSES, scanner FAILS. Dissolved." That was
> **overstated** — it tested only the fabrication half of #81 and ignored the failure mode #81's
> own filing text warns about: *"if a model ever fences its options list, skipping fenced content
> turns fabrication into total option loss — needs a ruling on which failure is preferred."*
> Tested below. The library **loses the entire list**. #81's done-when requires that *"a fenced
> options list is shown not to be silently emptied"* — the library **fails that half**.

| input | scanner | library |
|---|---|---|
| ```` ``` ```` fence with `- ` lines | `['Real option A', 'not_an_option: 1', 'also_not_an_option: 2', 'Real option B']` **FAIL** | `['Real option A', 'Real option B']` **PASS** |
| `~~~` fence + 4-space indented block | `['Real option', 'fabricated tilde']` **FAIL** | `['Real option']` **PASS** |

The fence never enters the list grammar at all — it is a sibling `fence` token holding the
`- ` lines as opaque `content`, so there is no list item to fabricate an option from:

```
bullet_list_open / list_item_open / ... 'Real option A' ... / bullet_list_close
fence  content='- not_an_option: 1\n- also_not_an_option: 2\n'
bullet_list_open / list_item_open / ... 'Real option B' ... / bullet_list_close
```

### The inversion — the half I initially failed to test

| input | scanner | library |
|---|---|---|
| **whole options list fenced** | `['Adopt PostgreSQL', 'Adopt SQLite', 'Adopt DuckDB']` **correct** | `[]` **TOTAL LOSS** |
| whole list fenced, ` ```markdown ` tag | `['Adopt PostgreSQL', 'Adopt SQLite']` **correct** | `[]` **TOTAL LOSS** |
| fenced list + one real option outside | `['Real option', 'fenced one', 'fenced two']` fabricates 2 | `['Real option']` correct |
| whole list indented 4sp | `[]` | `[]` — both lose it |

The scanner gets the fenced-whole-list case **right by accident** — its line-level blindness, the
very thing that causes the fabrication, is also what saves the payload. The library's block
awareness is strictly better on fabrication and strictly worse on loss.

**Which failure is preferred is exactly the ruling #81 asks for, and it is not mine to make.**
Note the asymmetry, though: fabrication yields a *plausibly wrong* option a consumer cannot
detect; total loss yields an honestly-empty `[]`, which #77's own doctrine calls readable
("an honest `[]` is readable as 'none extracted'; a fabricated item is not"). By that doctrine
the library's failure is the *safer* one — but it is still a regression against #81's done-when,
and the last row shows the scanner already has a partial version of the same loss bug.

**This is structural, not a heuristic** — and it covers tilde fences and indented code
blocks for free, which a scanner fix would have had to enumerate one at a time.

## #80 verdict — **rule-authority dissolved; the truncation choice remains ours**

Input:

```
- Adopt the library
  because it is spec-backed
  - Who endorsed it: sol
- Keep the scanner
```

| | result |
|---|---|
| scanner | `['Adopt the library', 'Keep the scanner']` — continuation **truncated** |
| library | `['Adopt the library because it is spec-backed', 'Keep the scanner']` |

**Continuation vs annotation IS a defined CommonMark rule, not a bespoke choice.** The token
tree makes the two structurally different kinds of thing:

- the continuation line is a **`softbreak` inside the same `inline` token** — lazy
  continuation, CommonMark §5.2/§4.8; it is the *same paragraph*.
- the annotation is a **nested `bullet_list_open` inside the `list_item`** — a different
  block.

```
list_item_open
  paragraph_open
    inline  content='Adopt the library\nbecause it is spec-backed'
      text 'Adopt the library' / softbreak / text 'because it is spec-backed'
  paragraph_close
  bullet_list_open          <-- annotation: structurally a child list
    list_item_open ... 'Who endorsed it: sol' ...
  bullet_list_close
list_item_close
```

So #80's open question — *is a continuation line part of the option?* — has a spec answer
(**yes**), and the scanner's current truncation is **against** it. What the library does
**not** decide is whether the delegation surface *wants* the full multi-line payload; that
stays a product ruling. The library reduces #80 from "invent a rule" to "apply or override a
known rule".

**Caveat:** rendering `softbreak` as `" "` inserts a space present in neither source *line*,
which trips the #77 `never_invents_characters` invariant as written (per-line). The fuzz test
passes only because its corpus has no multi-line items — latent, not clean.

## Hang / perf verdict — **hang dissolved; perf REGRESSES. This is the blocker.**

**Hang: PASS.** `- C:\Users\rob` / `- trailing\` → both implementations `['C:\\Users\\rob',
'trailing\\']` in ~0.1ms. The terra pass-4 infinite loop was a defect in our own
character-consuming loop; the library has no equivalent.

**Perf: FAIL.** `" *a" * n` (the terra pass-1 input):

| n | scanner | library | ratio |
|---|---|---|---|
| 2,000 | 5.9ms | 43.2ms | 7.3× |
| 4,000 | 13.3ms | 128.7ms | 9.7× |
| 8,000 | 41.4ms | 480.6ms | 11.6× |
| 16,000 | 123.7ms | 1,235.8ms | 10.0× |
| **30,000** | **122.9ms** | **3,786.0ms** | **30.8×** |

Isolating `md.parse()` from our flattening shows the cost is **inside the library**, and that
it degrades as input grows:

| n | `md.parse()` alone | growth vs previous |
|---|---|---|
| 8,000 | 416.6ms | — |
| 16,000 | 1,360.1ms | 3.27× (n^1.71) |
| 32,000 | 4,537.6ms | 3.34× (n^1.74) |
| 64,000 | **32,534.8ms** | **7.17× (n^2.85)** |

Our flattening roughly doubles the total but **scales identically** — it is not the cause.
markdown-it's inline emphasis delimiter processing is superlinear on this input, trending
quadratic-and-worse at scale. A model emitting one long option line is exactly the case #77's
bound was written for, and 30k chars already blows the 1.0s budget by ~3.8×.

The pass-2 close-and-open input (`"!_!*" * n`) is fine — ~linear (63/180/389/732ms for
2k/5k/10k/20k) and never breaches the bound. Only the pass-1 shape regresses.

**We cannot fix this in our layer.** The scanner's depth-capped single scan exists precisely
because this class of input is adversarial; adopting the library hands that guarantee back.

## Bottom line

| fork | library outcome |
|---|---|
| #81 fabrication half | **FIXED** — structural, and broader than a scanner fix |
| #81 loss half (fenced whole list) | **REGRESSED** — library returns `[]`; scanner is accidentally correct. #81's done-when unmet |
| #80 multi-line truncation | **rule-authority DISSOLVED** — spec-defined; product ruling remains |
| terra hang (`C:\Users\rob`) | **DISSOLVED** |
| terra perf (pass-1 shape) | **REGRESSED** — 30.8× slower at 30k, superlinear inside `md.parse()` |
| #77 unpaired-delimiter conservatism | **DIVERGES** — library is spec-correct, our test is not |

**RECOMMEND: KEEP-SCANNER** — and note #81 needs its *preferred-failure* ruling either way, since
neither implementation satisfies both halves of its done-when. Port markdown-it's
*fence-skipping structure* into the scanner **together with a guard for the fenced-whole-list
case** (e.g. fall back to the fenced content when fence-skipping would empty the section)
to close #81, and settle #80 by citing the CommonMark rule the library just proved exists;
do not adopt the parser while its `md.parse()` breaches the #77 latency bound we cannot patch.

## Reproduce

```
cd .claude/worktrees/markdown-it-py
py -m spike.evidence                                                   # decision cases + perf
py -m pytest tests/test_output.py -p spike.plugin_base -q              # scanner  120 pass
py -m pytest tests/test_output.py -p spike.plugin_swap -q              # library  118 pass
```

**Harness note (cost a false start — first diagnosis was wrong, corrected here):** a worktree
run can silently import the MAIN checkout's `src`, because the shared `.venv` editable-installs
`ai_council` from there.

I first concluded that the editable install's `__editable___ai_council_1_0_0_finder`
**MetaPathFinder outranks `PYTHONPATH`**. **That is false** — the finder is not even registered
in `sys.meta_path`, and `PYTHONPATH` does work. The real cause is **Git Bash / MSYS path
conversion**:

| `PYTHONPATH` value | resolves to |
|---|---|
| `/c/.../markdown-it-py/src` (single POSIX path) | **worktree** — MSYS auto-converts to `C:\...` |
| `/c/.../src;/tmp` (two POSIX paths, `;`-joined) | **MAIN** — conversion heuristic defeated, path unresolvable |
| `C:/.../src;C:/...` (Windows form) | **worktree** |

So from Git Bash, pass **Windows-form** paths or a **single** path. This does **not** affect any
result above: `spike/worktree_path.py` prepends the worktree `src` and then **asserts** the
resolved module path, and every run printed the worktree path before executing.
