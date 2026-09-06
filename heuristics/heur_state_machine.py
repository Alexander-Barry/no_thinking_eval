"""Shallow heuristics for state_machine: 5 states, permutations a-f, a word applied to the input state.

Rules: "a is given by: a(0) = 2, ..." and "Answer with a d e (x_0)", where the rightmost permutation is applied
first, so the order of application is the reverse of the written word. Every heuristic reads the start state and
applies at most two transitions, or reduces the word to symbol counts (order-blind closed forms).
"""
from __future__ import annotations

import re
from collections import Counter

FAMILY = "state_machine"


def _parse(item):
    """-> (transitions {(state, symbol): state}, symbols in order of application, start state, answer space)"""
    trans, seq = {}, None
    for r in item["rules"]:
        m = re.match(r"(\w) is given by: (.*)\.$", r)
        if m:
            for s, t in re.findall(r"\w\((\d+)\) = (\d+)", m.group(2)):
                trans[(s, m.group(1))] = t
        m = re.search(r"Answer with ([a-z ]+) \(x_0\)\.$", r)
        if m:
            seq = list(reversed(m.group(1).split()))
    return trans, seq, str(item["input"]).strip(), [str(a) for a in item["answer_space"]]


def _symbols(trans):
    return sorted({ch for _, ch in trans})


def _step(trans, s, ch):
    return trans.get((s, ch))


# ---- 0-step
def h_start(item):
    return str(item["input"]).strip()


# ---- 1-step
def h_first_step(item):
    """Apply only the first symbol to the start state."""
    trans, seq, s0, _ = _parse(item)
    return _step(trans, s0, seq[0])


def h_last_step(item):
    """Apply only the last symbol to the start state."""
    trans, seq, s0, _ = _parse(item)
    return _step(trans, s0, seq[-1])


def h_majority_symbol_step(item):
    """Apply the most frequent symbol once to the start state."""
    trans, seq, s0, _ = _parse(item)
    return _step(trans, s0, Counter(seq).most_common(1)[0][0])


def h_row_mode(item):
    """The most common image of the start state across all symbols (ties: first symbol)."""
    trans, seq, s0, _ = _parse(item)
    targets = [trans[(s0, ch)] for ch in _symbols(trans) if (s0, ch) in trans]
    return Counter(targets).most_common(1)[0][0] if targets else None


def h_exclude_one_step(item):
    """Rule out the one-step states (first and last symbol applied to the start) and answer the smallest survivor."""
    trans, seq, s0, space = _parse(item)
    bad = {_step(trans, s0, seq[0]), _step(trans, s0, seq[-1])}
    return next((a for a in space if a not in bad), None)


def h_exclude_one_step_and_start(item):
    """As above, also ruling out the start state."""
    trans, seq, s0, space = _parse(item)
    bad = {_step(trans, s0, seq[0]), _step(trans, s0, seq[-1]), s0}
    return next((a for a in space if a not in bad), None)


# ---- 2-step
def h_first_two_steps(item):
    trans, seq, s0, _ = _parse(item)
    s = _step(trans, s0, seq[0])
    return _step(trans, s, seq[1]) if len(seq) > 1 and s is not None else s


def h_last_two_from_start(item):
    """Apply the last two symbols to the start state."""
    trans, seq, s0, _ = _parse(item)
    if len(seq) < 2:
        return _step(trans, s0, seq[-1])
    s = _step(trans, s0, seq[-2])
    return _step(trans, s, seq[-1]) if s is not None else None


def h_first_then_last(item):
    trans, seq, s0, _ = _parse(item)
    s = _step(trans, s0, seq[0])
    return _step(trans, s, seq[-1]) if s is not None else None


def h_all_but_last(item):
    """Stop one symbol short of the end (a full simulation minus one step; the partial simulator's answer)."""
    trans, seq, s0, _ = _parse(item)
    s = s0
    for ch in seq[:-1]:
        s = _step(trans, s, ch)
    return s


# ---- order-blind closed forms from symbol counts
def _power(trans, s, ch, k):
    """ch applied k times to s, reduced modulo the length of s's cycle under ch (a lookup, not iteration)."""
    cyc = [s]
    x = _step(trans, s, ch)
    while x is not None and x != s and len(cyc) <= 6:
        cyc.append(x)
        x = _step(trans, x, ch)
    return cyc[k % len(cyc)]


def h_majority_symbol_power(item):
    """Ignore the minority symbols: the majority symbol applied as often as it occurs."""
    trans, seq, s0, _ = _parse(item)
    maj, k = Counter(seq).most_common(1)[0]
    return _power(trans, s0, maj, k)


def h_count_product(item):
    """Apply each symbol as often as it occurs, in alphabetical order (exact iff the symbols commute)."""
    trans, seq, s0, _ = _parse(item)
    c = Counter(seq)
    s = s0
    for ch in _symbols(trans):
        if c[ch]:
            s = _power(trans, s, ch, c[ch])
    return s


def heuristics():
    return {
        "start": h_start,
        "first_step": h_first_step,
        "last_step": h_last_step,
        "majority_symbol_step": h_majority_symbol_step,
        "row_mode": h_row_mode,
        "exclude_one_step_min": h_exclude_one_step,
        "exclude_one_step_and_start_min": h_exclude_one_step_and_start,
        "first_two_steps": h_first_two_steps,
        "last_two_from_start": h_last_two_from_start,
        "first_then_last": h_first_then_last,
        "all_but_last": h_all_but_last,
        "majority_symbol_power": h_majority_symbol_power,
        "count_product_order_blind": h_count_product,
    }
