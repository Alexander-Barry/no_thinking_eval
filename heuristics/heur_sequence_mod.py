"""Shallow heuristics for sequence_mod: term n = (f(term n-1) + term n-2) mod 10 with a stated permutation f.

Rules: "Term 1 is a. Term 2 is the input digit." / "Let f be the following permutation ...: f(0) = 7, ..." /
"Answer with term k+2." Every heuristic parses only the rules and the input: early terms, single reads of f,
arithmetic closed forms that ignore f, and stopping early.
"""
from __future__ import annotations

import re

FAMILY = "sequence_mod"


def _parse(it):
    """-> (term 1, f as dict, k = number of recurrence steps, input digit)"""
    text = "\n".join(it["rules"])
    a0 = int(re.search(r"Term 1 is (\d)", text).group(1))
    f = {int(a): int(b) for a, b in re.findall(r"f\((\d)\) = (\d)", text)}
    assert len(f) == 10, "permutation not parsed"
    k = int(re.search(r"Answer with term (\d+)", text).group(1)) - 2
    return a0, f, k, int(it["input"])


def _terms(a0, f, x, n):
    a, b = a0, x
    for _ in range(n):
        a, b = b, (f[b] + a) % 10
    return b


def input_itself(it):
    return it["input"]


def term_1(it):
    return str(_parse(it)[0])


def term_3(it):  # one step: f(input) + term 1
    a0, f, k, x = _parse(it)
    return str(_terms(a0, f, x, 1))


def term_4(it):  # two steps
    a0, f, k, x = _parse(it)
    return str(_terms(a0, f, x, 2))


def f_of_input(it):  # read f once, ignore the recurrence
    a0, f, k, x = _parse(it)
    return str(f[x])


def f_of_term1(it):
    a0, f, k, x = _parse(it)
    return str(f[a0])


def sum_mod10(it):  # (input + term 1) mod 10: the recurrence with f forgotten
    a0, f, k, x = _parse(it)
    return str((a0 + x) % 10)


def input_plus_5(it):  # the Fibonacci-mod-10 closed form at 20 steps
    return str((int(it["input"]) + 5) % 10)


def k_minus_1_steps(it):  # stop one term short
    a0, f, k, x = _parse(it)
    return str(_terms(a0, f, x, max(0, k - 1)))


def k_minus_2_steps(it):
    a0, f, k, x = _parse(it)
    return str(_terms(a0, f, x, max(0, k - 2)))


def f_iterated_k(it):  # iterate f alone on the input, forgetting the two-term memory
    a0, f, k, x = _parse(it)
    v = x
    for _ in range(k):
        v = f[v]
    return str(v)


def f_fixed_point(it):  # a digit f maps to itself, if any
    a0, f, k, x = _parse(it)
    fp = [d for d in range(10) if f[d] == d]
    return str(fp[0]) if fp else None


def parity_guess(it):  # 0 or 1 by the parity of input + term 1
    a0, f, k, x = _parse(it)
    return str((a0 + x) % 2)


def heuristics() -> dict:
    return {
        "input itself": input_itself, "term 1": term_1, "term 3 (one step)": term_3, "term 4 (two steps)": term_4,
        "f(input)": f_of_input, "f(term 1)": f_of_term1, "(input + term1) mod 10": sum_mod10,
        "input + 5": input_plus_5, "stop one term short": k_minus_1_steps, "stop two terms short": k_minus_2_steps,
        "iterate f k times on the input": f_iterated_k, "fixed point of f": f_fixed_point,
        "parity of input + term1": parity_guess,
    }
