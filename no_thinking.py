"""
Single-forward-pass rule tasks as an Inspect eval.

    inspect eval inspect_eval/no_thinking.py --model anthropic/claude-opus-5
    inspect eval inspect_eval/no_thinking.py --model openai/gpt-5.5 -T filler=1000
    inspect eval inspect_eval/no_thinking.py --model google/gemini-3-pro -T poem=true
    inspect eval inspect_eval/no_thinking.py --model anthropic/claude-haiku-4-5 -T families=chain_lookup,iterate_map -T levels=1,2,3

What is measured: whether a model can apply k in-context rules in ONE forward pass. Thinking is disabled
(reasoning_effort="none"), the answer budget is 4 tokens, and the FIRST visible token is the answer. Every item
has a single-token input and a single-token answer drawn uniformly from its answer space; nothing follows the
input line except optional filler tokens.

Task parameters (-T name=value):
  filler            number of one-token pause units appended after the input (0, 100, 1000 ...). Default 0.
  filler_position   "after" (default) or "before": before the Input line is the causal control (cannot see the input).
  filler_unit       the unit; " -" is one token per unit on every tokenizer tested. Default " -".
  poem              true = generated-filler condition: the trailer asks for a ~100-word poem about a dog before the
                    answer; the last output line is scored. Default false.
  addressee         name in the greeting ("Hello <name>, ..."). Default: inferred from the model id
                    (claude -> Claude, gemini/gemma -> Gemini, gpt/o-series -> ChatGPT); required for other providers.
  reasoning_effort  passed to the provider; "none" disables thinking on Claude 4.7+ and is a no-op on models without
                    thinking. Some OpenAI reasoning models only accept "minimal". Default "none".
  temperature       sampling temperature, or omit for the provider default (models that reject it must omit it).
  families, levels  comma-separated filters, e.g. families=chain_lookup,state_machine levels=1,2,3,4.
  data              path to the items file. Default: data/generated.jsonl next to this module.

Metrics are reported PER (family, level) CELL and never pooled across levels: a level is a depth on a grid shared by
all families (1 2 3 4 6 8 12 16 20 compositions), and families have different chance floors (0.04 letters, 0.10
digits, 0.20 five states). For each cell: accuracy, off_space (share of answers whose first token is not in the
answer space, i.e. the model started writing something else), and chance.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, GenerateConfig
from inspect_ai.scorer import (CORRECT, INCORRECT, Metric, SampleScore, Score, Scorer, Target, accuracy, grouped, metric,
                               scorer)
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data" / "generated.jsonl"

# ----------------------------------------------------------------------------------------------------------------
# Prompt helpers (the same functions as in generate.py; the wording itself lives in the data file)
# ----------------------------------------------------------------------------------------------------------------
ADDRESSEE = "Claude"  # the data is rendered greeting Claude; other providers get their own name at send time
ADDRESSEE_BY_PREFIX = [("claude", "Claude"), ("gemini", "Gemini"), ("gemma", "Gemini"),
                       ("gpt", "ChatGPT"), ("chatgpt", "ChatGPT"), ("o1", "ChatGPT"), ("o3", "ChatGPT"), ("o4", "ChatGPT")]
POEM_TRAILER = ("Before you answer, please write a short poem (around 100 words) about a dog. "
                "Then on a new line give just the answer as {phrase}, nothing else.")
_TRAILER_RE = re.compile(r"Please answer with just (.+?), nothing else, straight away, and without writing out any intermediate steps\.")


def addressee_for(model_name: str) -> str | None:
    m = model_name.lower().split("/")[-1]
    for prefix, name in ADDRESSEE_BY_PREFIX:
        if m.startswith(prefix):
            return name
    return None


def readdress(prompt: str, name: str) -> str:
    head = f"Hello {ADDRESSEE},"
    if not prompt.startswith(head):
        raise ValueError("prompt does not start with the expected greeting")
    return prompt if name == ADDRESSEE else f"Hello {name}," + prompt[len(head):]


def with_poem(prompt: str) -> str:
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
    """First whitespace-separated token, stripped of quotes and punctuation (the research scorer's rule)."""
    s = resp.strip().strip("\"'`*.:;,!()[]{} \n\t")
    tok = s.split()[0] if s.split() else ""
    tok = tok.strip("\"'`*.:;,!()[]{}")
    if tok in answer_space:
        return tok
    if tok.lower() in {a.lower() for a in answer_space}:
        return next(a for a in answer_space if a.lower() == tok.lower())
    return tok


# ----------------------------------------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------------------------------------
def load_dataset(path: Path, families: set[str] | None, levels: set[int] | None) -> MemoryDataset:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        level = it["difficulty"]["level"]
        if families and it["family"] not in families:
            continue
        if levels and level not in levels:
            continue
        knob = it["difficulty"].get("knob_value", "")
        samples.append(Sample(
            id=it["id"], input=it["prompt"], target=it["answer"],
            metadata={
                "family": it["family"], "level": level, "knob": it["difficulty"].get("knob", ""), "knob_value": knob,
                "cell": f"{it['family']} L{level} (depth {knob})",
                "answer_space": it["answer_space"], "chance": 1 / len(it["answer_space"]),
                "rationale": it["rationale"],
            }))
    if not samples:
        raise ValueError("no items selected")
    return MemoryDataset(samples, name="no_thinking")


# ----------------------------------------------------------------------------------------------------------------
# Solver: rewrite the prompt for this model and condition, then one generate call
# ----------------------------------------------------------------------------------------------------------------
@solver
def prepare_prompt(filler: int, filler_position: str, filler_unit: str, poem: bool, addressee: str | None) -> Solver:
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


# ----------------------------------------------------------------------------------------------------------------
# Scorer and per-cell metrics
# ----------------------------------------------------------------------------------------------------------------
@metric
def off_space() -> Metric:
    """Share of answers whose first token was not in the answer space (the model started writing something else)."""
    def compute(scores: list[SampleScore]) -> float:
        vals = [bool((s.score.metadata or {}).get("off_space", False)) for s in scores]
        return sum(vals) / len(vals) if vals else 0.0
    return compute


@metric
def chance() -> Metric:
    """The cell's chance floor (1 / answer-space size), so accuracy can be read against it."""
    def compute(scores: list[SampleScore]) -> float:
        vals = [float((s.sample_metadata or {}).get("chance", 0.0)) for s in scores]
        return sum(vals) / len(vals) if vals else 0.0
    return compute


@metric
def reasoning_blocks() -> Metric:
    """Share of samples whose output contained ANY reasoning/thinking content. Must be 0 for a single-forward-pass
    measurement; a non-zero value means the provider reasoned despite reasoning_effort='none'."""
    def compute(scores: list[SampleScore]) -> float:
        vals = [bool((s.score.metadata or {}).get("reasoning_blocks", 0)) for s in scores]
        return sum(vals) / len(vals) if vals else 0.0
    return compute


@metric
def output_tokens() -> Metric:
    """Mean output tokens per sample (includes any hidden reasoning tokens the provider bills): bounds hidden
    computation. Expect 1-4 for the token-budget condition (digit answers carry one invisible leading token)."""
    def compute(scores: list[SampleScore]) -> float:
        vals = [float((s.score.metadata or {}).get("output_tokens") or 0) for s in scores]
        return sum(vals) / len(vals) if vals else 0.0
    return compute


def _reasoning_block_count(state: TaskState) -> int:
    content = state.output.message.content if state.output and state.output.message else ""
    if isinstance(content, str):
        return 0
    return sum(1 for c in content if type(c).__name__ == "ContentReasoning")


@scorer(metrics=[grouped(accuracy(), "cell", all=False, name_template="{group_name} accuracy"),
                 grouped(off_space(), "cell", all=False, name_template="{group_name} off_space"),
                 grouped(chance(), "cell", all=False, name_template="{group_name} chance"),
                 grouped(reasoning_blocks(), "cell", all=False, name_template="{group_name} reasoning_blocks"),
                 grouped(output_tokens(), "cell", all=False, name_template="{group_name} output_tokens")])
def first_token(poem: bool) -> Scorer:
    """First visible token is the answer (last non-empty line in the poem condition). Besides the verdict, the score
    records everything needed to audit that no reasoning happened: reasoning block count, output token usage,
    stop reason, the raw completion and the greeting name used."""
    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion or ""
        text = completion
        if poem:  # everything up to the last non-empty line is the poem; the last line is the answer
            lines = [l for l in text.strip().splitlines() if l.strip()]
            text = lines[-1] if lines else ""
        space = list(state.metadata["answer_space"])
        tok = normalise(text, space)
        usage = state.output.usage
        return Score(value=CORRECT if tok == target.text else INCORRECT, answer=tok,
                     metadata={"off_space": tok not in space,
                               "reasoning_blocks": _reasoning_block_count(state),
                               "output_tokens": usage.output_tokens if usage else None,
                               "reasoning_tokens": getattr(usage, "reasoning_tokens", None) if usage else None,
                               "stop_reason": state.output.stop_reason,
                               "addressee": state.metadata.get("addressee"),
                               "raw": completion[:400] if poem else completion[:200]})
    return score


# ----------------------------------------------------------------------------------------------------------------
# Task
# ----------------------------------------------------------------------------------------------------------------
@task
def no_thinking(filler: int = 0, filler_position: str = "after", filler_unit: str = " -", poem: bool = False,
                addressee: str | None = None, reasoning_effort: str | None = "none", temperature: float | None = None,
                families: str | None = None, levels: str | None = None, data: str | None = None) -> Task:
    # -T values arrive as str, int or list depending on how they were typed ("1,2" / 3 / [1, 2])
    def as_list(v):
        if v is None:
            return None
        if isinstance(v, (list, tuple, set)):
            return [str(x) for x in v]
        return [x.strip() for x in str(v).split(",") if x.strip()]
    fams = set(as_list(families)) if families is not None else None
    lvls = {int(x) for x in as_list(levels)} if levels is not None else None
    dataset = load_dataset(Path(data) if data else DEFAULT_DATA, fams, lvls)
    config = GenerateConfig(max_tokens=400 if poem else 4, reasoning_effort=reasoning_effort, temperature=temperature)
    return Task(
        dataset=dataset,
        solver=[prepare_prompt(filler, filler_position, filler_unit, poem, addressee), generate()],
        scorer=first_token(poem),
        config=config,
        name="no_thinking",
    )
