"""Shallow heuristics for sequence_mod (table-driven two-term recurrence, v3.1 format).

Rules: "Term 1 is a. Term 2 is the input digit." / "This table maps every digit to another digit: 0 -> t0, ..." /
"... take the table's value for the term just before it, add the term before that, ... divided by 10." / "Answer with term k+2."
Every heuristic parses only the rules and the input. Written 2026-09-06 after the shortcut hunt replaced the linear
(Fibonacci) recurrence, whose closed form made the hardest rung a 0-step read.
"""
from __future__ import annotations

import re

FAMILY = "sequence_mod"


def _parse(it):
    text = "\n".join(it["rules"])
    a0 = int(re.search(r"Term 1 is (\d)", text).group(1))
    # v3.1 wording "f(0) = 7, ..." (signed off 2026-09-06); the earlier "0 -> 7" form is still accepted
    pairs = re.findall(r"f\((\d)\) = (\d)", text) or re.findall(r"(\d) -> (\d)", text)
    T = {int(k): int(v) for k, v in pairs}
    assert len(T) == 10, "permutation not parsed"
    k = int(re.search(r"Answer with term (\d+)", text).group(1)) - 2
    return a0, T, k, int(it["input"])


def _terms(a0, T, x, n):
    a, b = a0, x
    for _ in range(n):
        a, b = b, (T[b] + a) % 10
    return b


def input_itself(it):
    return it["input"]


def term_1(it):
    return str(_parse(it)[0])


def term_3(it):  # one real step: T[input] + term1
    a0, T, k, x = _parse(it)
    return str(_terms(a0, T, x, 1))


def term_4(it):  # two steps
    a0, T, k, x = _parse(it)
    return str(_terms(a0, T, x, 2))


def table_of_input(it):  # read the table once, ignore the recurrence
    a0, T, k, x = _parse(it)
    return str(T[x])


def table_of_term1(it):
    a0, T, k, x = _parse(it)
    return str(T[a0])


def sum_mod10(it):  # (input + term1) mod 10: the OLD linear recurrence's first step
    a0, T, k, x = _parse(it)
    return str((a0 + x) % 10)


def input_plus_5(it):  # the old top rung's identity
    return str((int(it["input"]) + 5) % 10)


def k_minus_1_steps(it):  # stop one term short
    a0, T, k, x = _parse(it)
    return str(_terms(a0, T, x, max(0, k - 1)))


def k_minus_2_steps(it):
    a0, T, k, x = _parse(it)
    return str(_terms(a0, T, x, max(0, k - 2)))


def table_iterated_k(it):  # treat it as iterate_map on the input, forgetting the two-term memory
    a0, T, k, x = _parse(it)
    v = x
    for _ in range(k):
        v = T[v]
    return str(v)


def table_fixed_point(it):  # a digit the table maps to itself, if any
    a0, T, k, x = _parse(it)
    fp = [d for d in range(10) if T[d] == d]
    return str(fp[0]) if fp else None


def parity_guess(it):  # smallest digit with the parity of (input + term1)
    a0, T, k, x = _parse(it)
    return str((a0 + x) % 2)


def heuristics() -> dict:
    return {
        "input itself": input_itself, "term 1": term_1, "term 3 (one step)": term_3, "term 4 (two steps)": term_4,
        "table[input]": table_of_input, "table[term 1]": table_of_term1, "(input + term1) mod 10": sum_mod10,
        "input + 5": input_plus_5, "stop one term short": k_minus_1_steps, "stop two terms short": k_minus_2_steps,
        "iterate the table k times on the input": table_iterated_k, "table fixed point": table_fixed_point,
        "parity of input + term1": parity_guess,
    }
