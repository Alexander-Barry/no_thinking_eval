# no_thinking: single-forward-pass rule tasks

An [Inspect](https://inspect.aisi.org.uk) eval that measures how much a model can reason in a single forward
pass, without generating any tokens first. Thinking is turned off and the answer is read from the first token
the model produces.

To make that measurable, every question has a known serial depth. The prompt sets out the rules of a small
system, such as a permutation of the alphabet or a five-state machine, and then gives a one-token input. The
answer is reached from the input in a fixed number of steps, each one application of the rules that needs the
result of the step before it. A model that can carry k such steps in one forward pass should be accurate up to
depth k and at chance beyond it.

| family | the system | one step | answers | depths |
|---|---|---|---|---|
| `iterate_map` | a permutation g of the capital letters, a single 26-cycle | apply g | A-Z, chance 0.04 | 1 2 3 4 6 8 12 |
| `state_machine` | 5 states and 6 permutations of them named a-f; the question gives a word of k letters | apply the next letter's permutation | 0-4, chance 0.20 | 1 2 3 4 6 8 12 16 20 |
| `chain_lookup` | k maps g_1..g_k of 10 entries each, word to word, the last one to digits | apply the next map | 0-9, chance 0.10 | 1 2 3 4 6 8 12 16 |
| `sequence_mod` | a sequence with term n = (f(term n-1) + term n-2) mod 10 for a stated permutation f | compute the next term | 0-9, chance 0.10 | 1 2 3 4 6 8 12 16 20 |

Every step is a bijection, so no step loses information and k steps never collapse into fewer. The depth grid
is shared, so level 5 means depth 6 in every family, and a family that cannot build a depth simply lacks that
level. Within each (family, level) cell the answers are balanced, so favouring a common answer earns nothing.
There are 40 items per cell, 33 cells and 1320 items in `data/generated.jsonl`.

## Run

```
pip install -r requirements.txt
inspect eval no_thinking.py --model anthropic/claude-opus-5
inspect eval no_thinking.py --model openai/gpt-5.5 -T filler=1000
inspect eval no_thinking.py --model google/gemini-3-pro -T poem=true
inspect eval no_thinking.py --model anthropic/claude-haiku-4-5 -T temperature=0 -T families=chain_lookup -T levels=1,2,3,4
```

## Settings and conditions

Thinking is turned off with Inspect's `reasoning_effort="none"`, which maps to each provider's thinking-off
setting and does nothing on models that have no thinking mode. The output budget is four tokens rather than one
only because some tokenizers emit a digit as two tokens. Only the first visible token is compared with the
answer. If it is not in the answer space, because the model began an explanation instead, the item is wrong and
is also counted in the cell's `off_space` rate, so failing and not complying can be told apart.

Nothing follows the input in the prompt, because every position after it is extra computation with access to
the input. The filler condition adds exactly that on purpose: N copies of `" -"`, one token each on every
tokenizer tested, appended after the input. The same filler placed before the input line is the control, as
those positions cannot see the input. The poem condition is filler the model generates itself: it is asked for
a short poem about a dog before the answer, the last line of its output is scored, and the budget rises to 400
tokens.

Task parameters, set with `-T name=value`:

| parameter | meaning | default |
|---|---|---|
| `filler` | how many filler units to add | 0 |
| `filler_position` | `after` the input, or `before` the Input line, the control | after |
| `filler_unit` | the unit repeated `filler` times | `" -"` |
| `poem` | `true` for the poem condition | false |
| `addressee` | the name in the greeting. The data greets Claude, and other models are greeted by their own name so that every model reads the same prompt apart from the name. Inferred from the model id for Claude, Gemini and ChatGPT models, required for anything else | inferred |
| `reasoning_effort` | passed to the provider; some OpenAI reasoning models only accept `minimal` | none |
| `temperature` | sampling temperature; must be omitted for models that reject it | omitted |
| `families`, `levels` | comma-separated filters | all |
| `data` | path to an items file | data/generated.jsonl |

## Reading the results

Results are reported per (family, level) cell, named like `chain_lookup L3 (depth 3)`, and never pooled across
levels or families. The cells differ in depth and in chance floor, so a pooled number would not mean anything.
Each cell reports:

- `accuracy`, to be read against the cell's `chance`.
- `off_space`, the share of answers whose first token was not in the answer space. These count as wrong. The
  rate separates a model that fails from one that does not comply.
- `reasoning_blocks`, the share of samples whose output contained any reasoning content. It must be 0. Anything
  else means the provider reasoned despite the setting, and the run is not a single-forward-pass measurement.
- `output_tokens`, the mean per sample including any hidden reasoning tokens the provider bills. Expect 1 to 4
  with the token budget and around 100 to 150 for the poem.

Each sample's score also stores the raw completion, the stop reason, the token usage and the greeting name, so a
run can be checked item by item in `inspect view`.

## What to expect (Claude models, September 2026)

Without filler the frontier Claude models manage about one step per forward pass, except on the word maps,
where Opus 5 is perfect at 2 to 3 hops and at 0.70 by 5. A hundred filler tokens buy one or two more steps, a
thousand a little more, and the poem is worth about a hundred filler tokens. Every family is at chance from
depth 12 for every model tried, with or without filler. The wrong answers on the deeper rungs are almost always
the right orbit at the wrong depth: the model stops one or two steps short, or overshoots by one.

## Files

- `no_thinking.py` is the Inspect task: dataset loading, the prompt rewriting for each model and condition, the
  first-token scorer and the per-cell metrics.
- `generate.py` holds the four generators, the prompt texts and the depth grid. `python generate.py --seed 0`
  rewrites `data/generated.jsonl`, and `--per-level` sets the items per cell.
- `data/generated.jsonl` has one item per line, described by `schema.json`: the prompt as sent, greeting Claude,
  the rules, the input, the answer and the answer space, the level and depth, and the path from input to answer.
- `audit.py` and `heuristics/` are the shortcut audit. Every heuristic, 13 to 24 per family, is scored against
  the gold answers per cell, and the best in each cell is written to `results/baselines.json`. After any change
  to a generator, run `python audit.py` and check that nothing beats chance on the cells deeper than what it
  computes. On earlier versions of these tasks, shortcuts accounted for whole cells: repeated symbols that
  returned to the same state, a linear recurrence with a closed form, a short cycle whose deep steps were a
  backward lookup.

Every prompt text, the header, the trailers and each family's rule wording, was signed off word for word by the
author. Don't change the wording without going back to them.
