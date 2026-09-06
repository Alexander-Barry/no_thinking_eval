"""Shallow-heuristic battery for the iterate_map family (one symbol->symbol table, a single cycle, applied k times).

Everything is parsed from the rendered rules / input / answer_space, so the same battery works for the current
digit version and for the planned capital-letter version as long as the table is rendered as "X -> Y, ...".
Positions ("row above / below", "+k") are indices into item["answer_space"], which is the key order of the table.

Heuristics fall in four groups:
  depth-j lookups      forward_j / backward_j      (backward_j == f^(N-j) on an N-cycle -> the known L6/L7 collapse)
  table-position cues  row neighbours of the input's row and of the row that maps TO the input
  arithmetic misreads  input +/- k, walk k rows, table applied to k, answer == k
  exclusion            lowest symbol not ruled out by the cycle structure (chance uplift, not a real shortcut)
"""
from __future__ import annotations

import re
from collections import Counter

FAMILY = "iterate_map"

_PAIR = re.compile(r"(\w+) -> (\w+)")
_STEPS = re.compile(r"exactly (\d+) time")


def _parse(item):
    """-> (f: dict sym->sym, finv, x0, k, order: list of symbols in table/answer-space order)"""
    f = {}
    # v3.1 wording (signed off 2026-09-06): "g(A) = E, ..." and "Answer with x_k"; the older "A -> E" / "exactly k times" still parse
    pairs = re.findall(r"g\((\w)\) = (\w)", item["rules"][0]) or _PAIR.findall(item["rules"][0])
    for a, b in pairs:
        f[a] = b
    finv = {b: a for a, b in f.items()}
    text = " ".join(item["rules"])
    m = re.search(r"Answer with x_(\d+)", text) or _STEPS.search(text)
    k = int(m.group(1)) if m else 1
    order = list(item["answer_space"])
    return f, finv, item["input"].strip(), k, order


def _walk(f, x, n):
    for _ in range(n):
        x = f.get(x)
        if x is None:
            return None
    return x


def _pos(order, x, d):
    """symbol d rows below x in the table (wraps)."""
    if x not in order:
        return None
    return order[(order.index(x) + d) % len(order)]


# ---- depth-j lookups -------------------------------------------------------------------------------------------
def input_itself(item):          # 0-step
    return _parse(item)[2]


def forward_1(item):
    f, _, x0, _, _ = _parse(item)
    return _walk(f, x0, 1)


def forward_2(item):
    f, _, x0, _, _ = _parse(item)
    return _walk(f, x0, 2)


def forward_3(item):
    f, _, x0, _, _ = _parse(item)
    return _walk(f, x0, 3)


def backward_1(item):            # "the symbol that maps TO the input"  (== f^9 on a 10-cycle)
    _, finv, x0, _, _ = _parse(item)
    return _walk(finv, x0, 1)


def backward_2(item):
    _, finv, x0, _, _ = _parse(item)
    return _walk(finv, x0, 2)


def backward_3(item):            # == f^7 on a 10-cycle
    _, finv, x0, _, _ = _parse(item)
    return _walk(finv, x0, 3)


# ---- table-position cues (table is listed in key order) ----------------------------------------------------------
def row_below_key(item):         # the key one row below the input's row  (input + 1)
    _, _, x0, _, order = _parse(item)
    return _pos(order, x0, 1)


def row_above_key(item):
    _, _, x0, _, order = _parse(item)
    return _pos(order, x0, -1)


def row_below_value(item):       # value in the row just below the input's row
    f, _, x0, _, order = _parse(item)
    return f.get(_pos(order, x0, 1))


def row_above_value(item):
    f, _, x0, _, order = _parse(item)
    return f.get(_pos(order, x0, -1))


def row_after_backward_value(item):   # value in the row just below the row whose value is the input
    f, finv, x0, _, order = _parse(item)
    return f.get(_pos(order, finv.get(x0), 1))


def first_row_value(item):
    f, _, _, _, order = _parse(item)
    return f.get(order[0])


def last_row_value(item):
    f, _, _, _, order = _parse(item)
    return f.get(order[-1])


# ---- arithmetic misreads of "apply k times" ------------------------------------------------------------------------
def key_plus_k(item):            # input + k  (Caesar-style)
    _, _, x0, k, order = _parse(item)
    return _pos(order, x0, k)


def key_minus_k(item):
    _, _, x0, k, order = _parse(item)
    return _pos(order, x0, -k)


def walk_k_rows_value(item):     # move k rows down from the input's row, read that value
    f, _, x0, k, order = _parse(item)
    return f.get(_pos(order, x0, k))


def walk_k1_rows_value(item):    # move k-1 rows down, read the value
    f, _, x0, k, order = _parse(item)
    return f.get(_pos(order, x0, k - 1))


def value_plus_k1(item):         # f(input) then slide k-1 symbols
    f, _, x0, k, order = _parse(item)
    return _pos(order, f.get(x0), k - 1)


def table_of_k(item):            # apply the table to the step count itself
    f, _, _, k, order = _parse(item)
    return f.get(str(k)) if str(k) in f else None


def answer_is_k(item):           # the step count digit appears one extra time in the prompt
    _, _, _, k, order = _parse(item)
    return str(k) if str(k) in order else None


# ---- textual cues ------------------------------------------------------------------------------------------------
def prompt_symbol_mode(item):    # most frequent answer-space symbol in the rendered prompt (ties -> first listed)
    _, _, _, _, order = _parse(item)
    body = item["prompt"].split("Rules:", 1)[-1]
    c = Counter(ch for ch in body if ch in set(order))
    if not c:
        return None
    top = max(c.values())
    return next(s for s in order if c.get(s) == top)


def prompt_symbol_mode_excl_input(item):
    _, _, x0, _, order = _parse(item)
    body = item["prompt"].split("Rules:", 1)[-1]
    c = Counter(ch for ch in body if ch in set(order) and ch != x0)
    if not c:
        return None
    top = max(c.values())
    return next(s for s in order if c.get(s) == top)


# ---- exclusion (chance uplift only) --------------------------------------------------------------------------------
def exclude_input_lowest(item):  # answer can never be the input on a single cycle: first symbol != input
    _, _, x0, _, order = _parse(item)
    return next((s for s in order if s != x0), None)


def exclude_depth2_lowest(item):
    """Rule out input and f^(+-1), f^(+-2) of the input unless k names one of them; answer the first survivor.
    Uses two forward + two backward lookups -> ~1/(N-5) instead of 1/N.  Structural note: valid because the
    generator guarantees a single N-cycle, which a model is NOT told."""
    f, finv, x0, k, order = _parse(item)
    n = len(order)
    bad = {x0}
    for j in (1, 2):
        if k % n != j:
            bad.add(_walk(f, x0, j))
        if k % n != (n - j) % n:
            bad.add(_walk(finv, x0, j))
    return next((s for s in order if s not in bad), None)


# ---- parity probe --------------------------------------------------------------------------------------------------
def parity_of_input_plus_k(item):   # first symbol whose index parity == parity(index(input)+k), excluding the input
    _, _, x0, k, order = _parse(item)
    if x0 not in order:
        return None
    want = (order.index(x0) + k) % 2
    return next((s for i, s in enumerate(order) if i % 2 == want and s != x0), None)


def heuristics() -> dict:
    return {
        "input_itself": input_itself,
        "forward_1": forward_1,
        "forward_2": forward_2,
        "forward_3": forward_3,
        "backward_1": backward_1,
        "backward_2": backward_2,
        "backward_3": backward_3,
        "row_below_key": row_below_key,
        "row_above_key": row_above_key,
        "row_below_value": row_below_value,
        "row_above_value": row_above_value,
        "row_after_backward_value": row_after_backward_value,
        "first_row_value": first_row_value,
        "last_row_value": last_row_value,
        "key_plus_k": key_plus_k,
        "key_minus_k": key_minus_k,
        "walk_k_rows_value": walk_k_rows_value,
        "walk_k1_rows_value": walk_k1_rows_value,
        "value_plus_k1": value_plus_k1,
        "table_of_k": table_of_k,
        "answer_is_k": answer_is_k,
        "prompt_symbol_mode": prompt_symbol_mode,
        "prompt_symbol_mode_excl_input": prompt_symbol_mode_excl_input,
        "exclude_input_lowest": exclude_input_lowest,
        "exclude_depth2_lowest": exclude_depth2_lowest,
        "parity_of_input_plus_k": parity_of_input_plus_k,
    }
