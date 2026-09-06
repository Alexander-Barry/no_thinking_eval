# no_thinking: single-forward-pass rule tasks

An [Inspect](https://inspect.aisi.org.uk) eval that measures how many in-context rule applications a model can
compose in **one forward pass**: thinking disabled, a 4-token answer budget, and the first visible token scored.
Every item states the rules of a small system, gives a single-token input, and asks for a single-token answer that
is uniformly distributed over the answer space by construction.

Four families, each a chain of **bijective** in-context maps whose length is the difficulty knob, so "k steps" is
never secretly fewer and no information is lost along the way:

| family | the system | answer space | depths |
|---|---|---|---|
| `iterate_map` | one permutation g of the capital letters (a single 26-cycle) applied k times | A-Z (chance 0.04) | 1 2 3 4 6 8 12 |
| `state_machine` | 5 states, 6 permutations named a-f, a word of k letters applied right to left (no immediate repeats) | 0-4 (chance 0.20) | 1 2 3 4 6 8 12 16 20 |
| `chain_lookup` | k maps g_1..g_k of 10 word -> word entries, the last to digits, disjoint vocabularies | 0-9 (chance 0.10) | 1 2 3 4 6 8 12 16 |
| `sequence_mod` | term n = (f(term n-1) + term n-2) mod 10 with a stated permutation f, k terms after the input | 0-9 (chance 0.10) | 1 2 3 4 6 8 12 16 20 |

A **level** is a depth on that shared grid (level 1 = depth 1, ..., level 9 = depth 20), so a level means the same
number of compositions in every family; a family simply lacks the levels whose depth it cannot construct.
40 items per (family, level) cell, 31 cells, 1320 items in `data/generated.jsonl`.

## Run

```
pip install -r requirements.txt
inspect eval no_thinking.py --model anthropic/claude-opus-5
inspect eval no_thinking.py --model openai/gpt-5.5 -T filler=1000
inspect eval no_thinking.py --model google/gemini-3-pro -T poem=true
inspect eval no_thinking.py --model anthropic/claude-haiku-4-5 -T temperature=0 -T families=chain_lookup -T levels=1,2,3,4
```

Task parameters (`-T name=value`):

| parameter | meaning | default |
|---|---|---|
| `filler` | pause tokens appended after the input: N copies of the one-token unit `" -"` | 0 |
| `filler_position` | `after` the input, or `before` the Input line (the causal control: filler that cannot see the input) | after |
| `poem` | generated-filler condition: the trailer asks for a ~100-word poem about a dog before the answer; the last output line is scored | false |
| `addressee` | name in the greeting ("Hello <name>, ..."); inferred for Claude / Gemini / ChatGPT model ids, required otherwise | inferred |
| `reasoning_effort` | passed to the provider; `none` disables thinking on Claude 4.7+ and is a no-op on models without thinking; some OpenAI reasoning models only accept `minimal` | none |
| `temperature` | sampling temperature; omit for models that reject it | provider default |
| `families`, `levels` | comma-separated filters | all |
| `data` | path to an items file | data/generated.jsonl |

Nothing ever follows the input line except filler tokens; every instruction, including the poem request, sits before it.

## Read the results

Metrics are reported **per cell** (`<family> L<level> (depth k)`) and never pooled across levels or families, because
the rungs differ in depth and in chance floor:

- `accuracy`: compare with the cell's `chance`.
- `off_space`: share of answers whose first token was not in the answer space (the model started writing its working,
  a tag, or prose). Counted as wrong.
- `reasoning_blocks`: share of samples whose output contained any reasoning content. **Must be 0**; otherwise the
  provider reasoned despite `reasoning_effort=none` and the run is not a single-forward-pass measurement.
- `output_tokens`: mean output tokens per sample, including any hidden reasoning tokens the provider bills. Expect 1-4
  in the token-budget conditions (digit answers carry one invisible leading token) and roughly 100-150 for the poem.

Each sample's score also stores the raw completion, the stop reason, the token usage and the greeting name used, so
a run can be audited item by item with `inspect view`.

## What to expect (Claude models, September 2026)

Without filler the frontier Claude models manage about one composition per forward pass, except on word maps, where
Opus 5 is perfect at 2-3 hops and 0.70 at 5. A hundred pause tokens buy one or two more compositions, a thousand a
little more, and the poem is worth about a hundred pause tokens. Every family is at chance from depth 12 for every
model tried, with or without filler. The wrong answers on the deeper rungs are almost always the right orbit at the
wrong depth: the model stops one or two compositions short, or overshoots by one.

## Regenerate or extend

```
python generate.py --out data/generated.jsonl --seed 0     # 40 items per cell; --per-level to change
python audit.py                                            # score the shallow heuristics against gold -> results/baselines.json
```

`generate.py` holds the four generators, the prompt texts and the depth grid; `audit.py` runs every heuristic in
`heuristics/` (13-26 per family: positional, 0-step and 1-step, closed-form and textual cues) against the gold
answers per cell. After any change to a generator, run the audit and check that every heuristic stays at chance on
the deeper rungs before spending API calls: on earlier versions of these tasks, shortcuts (repeated-symbol
coincidences, a linear recurrence's closed form, a short cycle's backward lookup) accounted for whole cells.

Every prompt text (the two-paragraph header, the per-family trailers, the poem trailer, each family's rule wording)
was signed off by the author. Do not edit the wording without going back to them.
