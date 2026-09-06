"""
Single-forward-pass rule tasks.

This eval measures how much a model can reason in a single forward pass, without generating any tokens
first. Thinking is turned off and the answer is read from the first token the model produces.

To make that measurable, every question has a known serial depth. The prompt sets out the rules of a small
system, such as a permutation of the alphabet or a five-state machine, and then gives a one-token input. The
answer is reached from the input in a fixed number of steps, each one application of the rules that needs the
result of the step before it. A model that can carry k such steps in one forward pass should be accurate up
to depth k and at chance beyond it. There are four families of rules on a shared grid of depths from 1 to 20,
each built so that the k steps never collapse into fewer.

Thinking is turned off with Inspect's reasoning_effort="none", which maps to each provider's thinking-off
setting and does nothing on models that have no thinking mode. The output budget is four tokens rather than
one only because some tokenizers emit a digit as two tokens. Only the first visible token is compared with
the answer. If it is not in the answer space, because the model began an explanation instead, the item is
wrong and is also counted in the cell's off_space rate, so failing and not complying can be told apart.

Nothing follows the input in the prompt, because every position after it is extra computation with access to
the input. The filler condition adds exactly that on purpose: N copies of " -", one token each on every
tokenizer tested, appended after the input. The same filler placed before the input line is the control, as
those positions cannot see the input. The poem condition is filler the model generates itself: it is asked
for a short poem about a dog before the answer, the last line of its output is scored, and the budget rises
to 400 tokens. Each score also records the number of reasoning blocks, the output token count and the stop
reason, so a run can be checked afterwards for reasoning that slipped through.

Results are reported per (family, level) cell and never pooled, because the cells differ in depth and in
chance floor: 1/26 for letters, 1/10 for digits, 1/5 for the five states. A level is a position on the grid
1 2 3 4 6 8 12 16 20, so level 5 means depth 6 in every family, and a family that cannot build a depth simply
lacks that level. Within each cell the answers are balanced, so favouring a common answer earns nothing.

Usage:

    inspect eval no_thinking.py --model anthropic/claude-opus-5
    inspect eval no_thinking.py --model openai/gpt-5.5 -T filler=1000
    inspect eval no_thinking.py --model google/gemini-3-pro -T poem=true
    inspect eval no_thinking.py --model anthropic/claude-haiku-4-5 -T families=chain_lookup,iterate_map -T levels=1,2,3

Task parameters, set with -T name=value:

  filler            How many filler units to add. Default 0.
  filler_position   "after" the input (default) or "before" the Input line, the control.
  filler_unit       The unit repeated filler times. Default " -".
  poem              true for the poem condition. Default false.
  addressee         The name in the greeting. The data greets Claude, and other models are greeted by their
                    own name so that every model reads the same prompt apart from the name. Inferred from the
                    model id for Claude, Gemini and ChatGPT models, required for anything else.
  reasoning_effort  Passed to the provider. Default "none". Some OpenAI reasoning models only accept "minimal".
  temperature       Sampling temperature. Omitted by default, and must be omitted for models that reject it.
  families, levels  Comma-separated filters, e.g. families=chain_lookup,state_machine levels=1,2,3,4.
  data              Path to an items file. Default: data/generated.jsonl next to this module.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, ContentReasoning, GenerateConfig
from inspect_ai.scorer import (CORRECT, INCORRECT, Metric, SampleScore, Score, Scorer, Target, accuracy, grouped, metric,
                               scorer)
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver

from generate import HEADER, POEM_TRAILER, TRAILER

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data" / "generated.jsonl"

# The data greets Claude; a model from another provider gets its own name at send time, so every model reads the
# same prompt apart from the name.
GREETING = "Hello Claude,"
assert HEADER.startswith(GREETING)
ADDRESSEE_BY_PREFIX = [("claude", "Claude"), ("gemini", "Gemini"), ("gemma", "Gemini"),
                       ("gpt", "ChatGPT"), ("chatgpt", "ChatGPT"), ("o1", "ChatGPT"), ("o3", "ChatGPT"), ("o4", "ChatGPT")]
_TRAILER_RE = re.compile(re.escape(TRAILER).replace(re.escape("{phrase}"), "(.+?)"))


def addressee_for(model_name: str) -> str | None:
    m = model_name.lower().split("/")[-1]
    for prefix, name in ADDRESSEE_BY_PREFIX:
        if m.startswith(prefix):
            return name
    return None


def readdress(prompt: str, name: str) -> str:
    if not prompt.startswith(GREETING):
        raise ValueError("prompt does not start with the expected greeting")
    return f"Hello {name}," + prompt[len(GREETING):]


def with_poem(prompt: str) -> str:
    """The same prompt with the answer trailer replaced by the poem trailer (same answer phrase)."""
    m = _TRAILER_RE.search(prompt)
    if not m:
        raise ValueError("prompt has no recognised answer trailer")
    return prompt[:m.start()] + POEM_TRAILER.format(phrase=m.group(1)) + prompt[m.end():]


def with_filler(prompt: str, n: int, unit: str, position: str) -> str:
    if n <= 0:
        return prompt
    if position == "after":
        return prompt + "\n\n" + unit * n
    head, sep, tail = prompt.rpartition("\nInput: ")
    if not sep:
        raise ValueError("prompt has no Input line")
    return head.rstrip("\n") + "\n\n" + unit * n + "\n\nInput: " + tail


def normalise(resp: str, answer_space: list[str]) -> str:
    """The first whitespace-separated token, stripped of quotes and punctuation and case-folded into the answer space."""
    s = resp.strip().strip("\"'`*.:;,!()[]{} \n\t")
    tok = s.split()[0] if s.split() else ""
    tok = tok.strip("\"'`*.:;,!()[]{}")
    if tok in answer_space:
        return tok
    if tok.lower() in {a.lower() for a in answer_space}:
        return next(a for a in answer_space if a.lower() == tok.lower())
    return tok


def load_dataset(path: Path, families: set[str] | None, levels: set[int] | None) -> MemoryDataset:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        level, depth = it["difficulty"]["level"], it["difficulty"]["depth"]
        if families and it["family"] not in families:
            continue
        if levels and level not in levels:
            continue
        samples.append(Sample(
            id=it["id"], input=it["prompt"], target=it["answer"],
            metadata={
                "family": it["family"], "level": level, "depth": depth,
                "cell": f"{it['family']} L{level} (depth {depth})",
                "answer_space": it["answer_space"], "chance": 1 / len(it["answer_space"]),
                "rationale": it["rationale"],
            }))
    if not samples:
        raise ValueError("no items selected")
    return MemoryDataset(samples, name="no_thinking")


@solver
def prepare_prompt(filler: int, filler_position: str, filler_unit: str, poem: bool, addressee: str | None) -> Solver:
    """Rewrite the stored prompt for this model and condition: greeting name, poem trailer, filler."""
    async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
        name = addressee or addressee_for(state.model.name)
        if not name:
            raise ValueError(f"cannot infer the greeting name for model '{state.model}'; pass -T addressee=<Name>")
        prompt = readdress(state.input_text, name)
        if poem:
            prompt = with_poem(prompt)
        prompt = with_filler(prompt, filler, filler_unit, filler_position)
        state.messages = [ChatMessageUser(content=prompt)]
        state.metadata["addressee"] = name
        return state
    return solve


def _mean_of(scores: list[SampleScore], key: str, from_sample: bool = False) -> float:
    vals = [float(((s.sample_metadata if from_sample else s.score.metadata) or {}).get(key) or 0) for s in scores]
    return sum(vals) / len(vals) if vals else 0.0


@metric
def off_space() -> Metric:
    """Share of answers whose first token was not in the answer space: the model began something other than the answer."""
    return lambda scores: _mean_of(scores, "off_space")


@metric
def chance() -> Metric:
    """The cell's chance floor, one over the size of the answer space, so accuracy can be read against it."""
    return lambda scores: _mean_of(scores, "chance", from_sample=True)


@metric
def reasoning_blocks() -> Metric:
    """Share of samples whose output contained any reasoning content. It must be 0. Anything else means the provider
    reasoned despite the setting, and the run is not a single-forward-pass measurement."""
    return lambda scores: _mean_of(scores, "reasoned")


@metric
def output_tokens() -> Metric:
    """Mean output tokens per sample, including hidden reasoning tokens the provider bills, which bounds hidden
    computation. Expect 1 to 4 with the token budget, since a digit can cost two tokens."""
    return lambda scores: _mean_of(scores, "output_tokens")


@scorer(metrics=[grouped(accuracy(), "cell", all=False, name_template="{group_name} accuracy"),
                 grouped(off_space(), "cell", all=False, name_template="{group_name} off_space"),
                 grouped(chance(), "cell", all=False, name_template="{group_name} chance"),
                 grouped(reasoning_blocks(), "cell", all=False, name_template="{group_name} reasoning_blocks"),
                 grouped(output_tokens(), "cell", all=False, name_template="{group_name} output_tokens")])
def first_token(poem: bool) -> Scorer:
    """Compares the first visible token with the answer, or the last non-empty line in the poem condition. The score
    also records what is needed to check that no reasoning happened: the reasoning block count, the token usage,
    the stop reason, the raw completion and the greeting name."""
    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion or ""
        text = completion
        if poem:
            lines = [l for l in text.strip().splitlines() if l.strip()]
            text = lines[-1] if lines else ""
        space = list(state.metadata["answer_space"])
        tok = normalise(text, space)
        content = state.output.message.content
        n_reasoning = 0 if isinstance(content, str) else sum(isinstance(c, ContentReasoning) for c in content)
        usage = state.output.usage
        return Score(value=CORRECT if tok == target.text else INCORRECT, answer=tok,
                     metadata={"off_space": tok not in space,
                               "reasoning_blocks": n_reasoning,
                               "reasoned": n_reasoning > 0,
                               "output_tokens": usage.output_tokens if usage else None,
                               "reasoning_tokens": usage.reasoning_tokens if usage else None,
                               "stop_reason": state.output.stop_reason,
                               "addressee": state.metadata.get("addressee"),
                               "raw": completion[:400] if poem else completion[:200]})
    return score


@task
def no_thinking(filler: int = 0, filler_position: str = "after", filler_unit: str = " -", poem: bool = False,
                addressee: str | None = None, reasoning_effort: str | None = "none", temperature: float | None = None,
                families: str | None = None, levels: str | None = None, data: str | None = None) -> Task:
    def as_list(v) -> list[str]:  # -T values arrive as str, int or list depending on how they were typed
        if isinstance(v, (list, tuple, set)):
            return [str(x) for x in v]
        return [x.strip() for x in str(v).split(",") if x.strip()]
    fams = set(as_list(families)) if families is not None else None
    lvls = {int(x) for x in as_list(levels)} if levels is not None else None
    return Task(
        dataset=load_dataset(Path(data) if data else DEFAULT_DATA, fams, lvls),
        solver=[prepare_prompt(filler, filler_position, filler_unit, poem, addressee), generate()],
        scorer=first_token(poem),
        config=GenerateConfig(max_tokens=400 if poem else 4, reasoning_effort=reasoning_effort, temperature=temperature),
        name="no_thinking",
    )
