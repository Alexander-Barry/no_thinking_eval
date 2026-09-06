"""Shallow heuristics for iterate_map: one permutation g of the letters (a single 26-cycle) applied k times.

Everything is parsed from the rules, the input and the answer space. "Row" means position in the listing order of
g, which is alphabetical, so row arithmetic is also letter arithmetic (a Caesar shift).

Groups: depth-j lookups forward and backward; listing-position cues; misreadings of "apply k times" as arithmetic;
textual frequency; exclusion of letters the cycle structure rules out (a chance uplift rather than a shortcut).
"""
from __future__ import annotations

import re
from collections import Counter

FAMILY = "iterate_map"


def _parse(item):
    """-> (g, g inverse, input letter, k, listing order)"""
    g = dict(re.findall(r"g\((\w)\) = (\w)", item["rules"][0]))
    m = re.search(r"exactly (\d+) time", " ".join(item["rules"]))
    return g, {v: k for k, v in g.items()}, item["input"].strip(), int(m.group(1)) if m else 1, list(item["answer_space"])


def _walk(f, x, n):
    for _ in range(n):
        x = f.get(x)
        if x is None:
            return None
    return x


def _pos(order, x, d):
    """The letter d rows below x in the listing (wrapping)."""
    if x not in order:
        return None
    return order[(order.index(x) + d) % len(order)]


# ---- depth-j lookups
def input_itself(item):
    return _parse(item)[2]


def forward_1(item):
    g, _, x0, _, _ = _parse(item)
    return _walk(g, x0, 1)


def forward_2(item):
    g, _, x0, _, _ = _parse(item)
    return _walk(g, x0, 2)


def forward_3(item):
    g, _, x0, _, _ = _parse(item)
    return _walk(g, x0, 3)


def backward_1(item):  # the letter that maps TO the input (g^-1 = g^25 on a 26-cycle)
    _, ginv, x0, _, _ = _parse(item)
    return _walk(ginv, x0, 1)


def backward_2(item):
    _, ginv, x0, _, _ = _parse(item)
    return _walk(ginv, x0, 2)


def backward_3(item):
    _, ginv, x0, _, _ = _parse(item)
    return _walk(ginv, x0, 3)


# ---- listing-position cues
def row_below_key(item):  # the letter after the input in the listing (input + 1)
    _, _, x0, _, order = _parse(item)
    return _pos(order, x0, 1)


def row_above_key(item):
    _, _, x0, _, order = _parse(item)
    return _pos(order, x0, -1)


def row_below_value(item):  # the value listed just after the input's entry
    g, _, x0, _, order = _parse(item)
    return g.get(_pos(order, x0, 1))


def row_above_value(item):
    g, _, x0, _, order = _parse(item)
    return g.get(_pos(order, x0, -1))


def row_after_backward_value(item):  # the value listed just after the entry whose value is the input
    g, ginv, x0, _, order = _parse(item)
    return g.get(_pos(order, ginv.get(x0), 1))


def first_row_value(item):
    g, _, _, _, order = _parse(item)
    return g.get(order[0])


def last_row_value(item):
    g, _, _, _, order = _parse(item)
    return g.get(order[-1])


# ---- "apply k times" misread as arithmetic
def key_plus_k(item):  # input + k (Caesar shift)
    _, _, x0, k, order = _parse(item)
    return _pos(order, x0, k)


def key_minus_k(item):
    _, _, x0, k, order = _parse(item)
    return _pos(order, x0, -k)


def walk_k_rows_value(item):  # the value listed k entries after the input's
    g, _, x0, k, order = _parse(item)
    return g.get(_pos(order, x0, k))


def walk_k1_rows_value(item):
    g, _, x0, k, order = _parse(item)
    return g.get(_pos(order, x0, k - 1))


def value_plus_k1(item):  # g(input) then slide k-1 letters
    g, _, x0, k, order = _parse(item)
    return _pos(order, g.get(x0), k - 1)


# ---- textual frequency
def prompt_symbol_mode(item):  # the most frequent capital letter after "Rules:" (ties: alphabetically first)
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


# ---- exclusion (chance uplift only)
def exclude_input_lowest(item):  # on a single cycle the answer is never the input: the first other letter
    _, _, x0, _, order = _parse(item)
    return next((s for s in order if s != x0), None)


def exclude_depth2_lowest(item):
    """Rule out the input and g^(+-1), g^(+-2) of it unless k names one of them, then answer the first survivor.
    Valid only because the generator guarantees a single cycle, which the model is not told."""
    g, ginv, x0, k, order = _parse(item)
    n = len(order)
    bad = {x0}
    for j in (1, 2):
        if k % n != j:
            bad.add(_walk(g, x0, j))
        if k % n != (n - j) % n:
            bad.add(_walk(ginv, x0, j))
    return next((s for s in order if s not in bad), None)


def parity_of_input_plus_k(item):  # the first letter (other than the input) whose index has the parity of index + k
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
        "prompt_symbol_mode": prompt_symbol_mode,
        "prompt_symbol_mode_excl_input": prompt_symbol_mode_excl_input,
        "exclude_input_lowest": exclude_input_lowest,
        "exclude_depth2_lowest": exclude_depth2_lowest,
        "parity_of_input_plus_k": parity_of_input_plus_k,
    }
