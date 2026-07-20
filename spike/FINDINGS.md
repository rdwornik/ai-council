# SPIKE — markdown-it-py vs the hand-written options scanner

**Date:** 2026-07-20 · **Worktree:** `spike-md-parser` · **Status:** THROWAWAY — nothing merges
**Governance:** #77 (merged extractor contract, `94421c2`), #80, #81
**Baseline commit:** `94421c2` · `src/ai_council/output.py` NOT modified · scanner NOT deleted

## Recommendation

**KEEP-SCANNER.** The library dissolves #81 outright and dissolves #80's *rule-authority*
question, but it **reintroduces the performance fork the scanner needed 6 terra passes to
close** — and that regression lives inside `md.parse()`, where we cannot fix it.

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

## #81 verdict — **library PASSES, scanner FAILS. Dissolved.**

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
| #81 fenced-block fabrication | **DISSOLVED** — structural, and broader than a scanner fix |
| #80 multi-line truncation | **rule-authority DISSOLVED** — spec-defined; product ruling remains |
| terra hang (`C:\Users\rob`) | **DISSOLVED** |
| terra perf (pass-1 shape) | **REGRESSED** — 30.8× slower at 30k, superlinear inside `md.parse()` |
| #77 unpaired-delimiter conservatism | **DIVERGES** — library is spec-correct, our test is not |

**RECOMMEND: KEEP-SCANNER** — port markdown-it's *fence-skipping structure* into the scanner
to close #81, and settle #80 by citing the CommonMark rule the library just proved exists;
do not adopt the parser while its `md.parse()` breaches the #77 latency bound we cannot patch.

## Reproduce

```
cd .claude/worktrees/markdown-it-py
py -m spike.evidence                                                   # decision cases + perf
py -m pytest tests/test_output.py -p spike.plugin_base -q              # scanner  120 pass
py -m pytest tests/test_output.py -p spike.plugin_swap -q              # library  118 pass
```

**Harness note (cost us a false start):** the shared `.venv` editable install ships
`__editable___ai_council_1_0_0_finder`, a **MetaPathFinder** pinned to the MAIN checkout's
`src/`. A MetaPathFinder is consulted **before `sys.path`**, so `PYTHONPATH` alone does **not**
redirect the import — a worktree run silently tests the main checkout. `spike/worktree_path.py`
drops the finder, re-imports, and **asserts** the resolved path.
