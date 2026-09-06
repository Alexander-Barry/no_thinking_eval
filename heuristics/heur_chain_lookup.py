"""Shallow heuristics for chain_lookup: k maps g_1..g_k of ten entries each over disjoint word layers, the last to digits.

Rules: "g_1 is given by: g_1(dog) = red, ..." per map. Each map is listed in a random key order and the entries
are shuffled independently, so a row index carries no information across maps; the heuristics below check that
(row alignment after 0, 1 or 2 lookups), plus constant, positional, alphabetical and frequency cues.
"""
from __future__ import annotations

import re
from collections import Counter

FAMILY = "chain_lookup"


def parse(item):
    """-> the maps in prompt order, each an ordered list of (key, value) pairs."""
    tables = []
    for r in item["rules"]:
        m = re.match(r"^g_\d+ is given by: (.*)\.$", r.strip())
        if m:
            tables.append(re.findall(r"g_\d+\((\w+)\) = (\w+)", m.group(1)))
    return tables


def keys(t):
    return [k for k, _ in t]


def vals(t):
    return [v for _, v in t]


def lookup(t, w):
    return dict(t).get(w)


def row_of_key(t, w):
    ks = keys(t)
    return ks.index(w) if w in ks else None


def row_of_val(t, w):
    vs = vals(t)
    return vs.index(w) if w in vs else None


def digit_at_row(last, r):
    return vals(last)[r] if r is not None and 0 <= r < len(last) else None


def hop(item, T, n):
    """The word reached after n forward lookups from the input, or None if the chain breaks."""
    w = item["input"]
    for h in range(n):
        w = lookup(T[h], w)
        if w is None:
            return None
    return w


# ---- constants and positions in the last map
def h_last_first(it):
    return vals(parse(it)[-1])[0]


def h_last_last(it):
    return vals(parse(it)[-1])[-1]


def h_last_middle(it):
    T = parse(it)
    return vals(T[-1])[len(T[-1]) // 2]


# ---- row alignment after 0, 1 or 2 lookups (the full solution when the depth equals the lookups)
def h_row_input(it):
    """The digit at the input's row of g_1, read in the last map."""
    T = parse(it)
    return digit_at_row(T[-1], row_of_key(T[0], it["input"]))


def h_row_input_mirror(it):
    T = parse(it)
    r = row_of_key(T[0], it["input"])
    return digit_at_row(T[-1], len(T[-1]) - 1 - r) if r is not None else None


def h_row_hop1(it):
    """One lookup, then the row of the reached word among g_2's keys, read in the last map."""
    T = parse(it)
    if len(T) < 2:
        return None
    return digit_at_row(T[-1], row_of_key(T[1], hop(it, T, 1)))


def h_row_hop1_valpos(it):
    """One lookup, then the row of the reached word among g_1's values (the input's own row)."""
    T = parse(it)
    return digit_at_row(T[-1], row_of_val(T[0], hop(it, T, 1)))


def h_row_hop2(it):
    T = parse(it)
    if len(T) < 3:
        return None
    return digit_at_row(T[-1], row_of_key(T[2], hop(it, T, 2)))


def h_row_hop2_valpos(it):
    T = parse(it)
    if len(T) < 3:
        return None
    return digit_at_row(T[-1], row_of_val(T[1], hop(it, T, 2)))


# ---- a row index or rank read as the digit itself
def h_input_row_digit(it):
    r = row_of_key(parse(it)[0], it["input"])
    return str(r) if r is not None else None


def h_hop1_row_digit(it):
    T = parse(it)
    r = row_of_key(T[1], hop(it, T, 1)) if len(T) > 1 else None
    return str(r) if r is not None else None


def h_alpha_rank_row(it):
    """The alphabetical rank of the input among g_1's keys, as a row of the last map."""
    T = parse(it)
    return digit_at_row(T[-1], sorted(keys(T[0])).index(it["input"]))


def h_alpha_rank_digit(it):
    return str(sorted(keys(parse(it)[0])).index(it["input"]))


# ---- textual cues
def h_prompt_digit_freq(it):
    """The most frequent digit character in the prompt (map indices included); ties to the smallest."""
    c = Counter(ch for ch in it["prompt"] if ch.isdigit())
    m = max(c.values())
    return min(d for d, n in c.items() if n == m)


def h_hops_mod10(it):
    return str(len(parse(it)) % 10)


# ---- one lookup from the end
def h_end_first_value(it):
    """The first value listed in the penultimate map, looked up in the last map."""
    T = parse(it)
    return lookup(T[-1], vals(T[-2])[0]) if len(T) >= 2 else None


def h_end_last_value(it):
    T = parse(it)
    return lookup(T[-1], vals(T[-2])[-1]) if len(T) >= 2 else None


def h_end_row_input(it):
    """The value at the input's row of the penultimate map, looked up in the last map."""
    T = parse(it)
    r = row_of_key(T[0], it["input"])
    return lookup(T[-1], vals(T[-2])[r]) if len(T) >= 2 and r is not None else None


def h_last_alpha_first_key(it):
    T = parse(it)
    return lookup(T[-1], min(keys(T[-1])))


def h_last_alpha_last_key(it):
    T = parse(it)
    return lookup(T[-1], max(keys(T[-1])))


def heuristics():
    return {
        "last_map_first_digit": h_last_first,
        "last_map_last_digit": h_last_last,
        "last_map_middle_digit": h_last_middle,
        "row_align_input": h_row_input,
        "row_align_input_mirror": h_row_input_mirror,
        "hop1_then_row_align_keys": h_row_hop1,
        "hop1_then_row_align_values": h_row_hop1_valpos,
        "hop2_then_row_align_keys": h_row_hop2,
        "hop2_then_row_align_values": h_row_hop2_valpos,
        "input_row_as_digit": h_input_row_digit,
        "hop1_row_as_digit": h_hop1_row_digit,
        "alpha_rank_row_align": h_alpha_rank_row,
        "alpha_rank_as_digit": h_alpha_rank_digit,
        "prompt_digit_frequency": h_prompt_digit_freq,
        "hops_mod_10": h_hops_mod10,
        "end_first_value_1hop": h_end_first_value,
        "end_last_value_1hop": h_end_last_value,
        "end_row_align_1hop": h_end_row_input,
        "last_alpha_first_key": h_last_alpha_first_key,
        "last_alpha_last_key": h_last_alpha_last_key,
    }
