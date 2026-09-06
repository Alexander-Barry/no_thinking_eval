"""Shallow heuristics for the state_machine family (5 states, alphabet abc, each symbol a fixed-point-free
permutation; input = start state; the symbol string lives in the rules).

Every heuristic in heuristics() is something a model could do WITHOUT iterating over the string: read the
start state, apply at most one or two transitions, count symbols, or exploit the fact that the symbol
permutations have no fixed points. Structural checks (which use the full simulation) are NOT in heuristics();
they live in structural_checks() so the audit's "best shallow heuristic" is not polluted by a deep computation.
"""
import re
from collections import Counter

FAMILY = "state_machine"


# ---------------------------------------------------------------- parsing (rules / input / answer_space only)
def _parse(item):
    trans = {}
    seq = None
    for r in item["rules"]:
        # v3.1 wording (signed off 2026-09-06): "a is given by: a(0) = 2, ..." and "Answer with a d e (x_0)", where the
        # rightmost permutation is applied first, so the application sequence is the reverse of the written word
        m = re.match(r"(\w) is given by: (.*)\.$", r)
        if m:
            ch = m.group(1)
            for s, t in re.findall(r"\w\((\d+)\) = (\d+)", m.group(2)):
                trans[(s, ch)] = t
        m = re.search(r"Answer with ([a-z ]+) \(x_0\)\.$", r)
        if m:
            seq = list(reversed(m.group(1).split()))
        # older wording
        m = re.match(r"In state (\d+), (.*)\.$", r)
        if m:
            s = m.group(1)
            for ch, t in re.findall(r"on '(\w)' go to state (\d+)", m.group(2)):
                trans[(s, ch)] = t
        m = re.search(r"one at a time: ([a-z ]+)\.$", r)
        if m:
            seq = m.group(1).split()
    return trans, seq, str(item["input"]).strip(), [str(a) for a in item["answer_space"]]


def _step(trans, s, ch):
    return trans.get((s, ch))


# ---------------------------------------------------------------- 0-step
def h_start(item):
    """Answer = the start state (0 steps)."""
    return str(item["input"]).strip()


def h_last_option(item):
    return str(item["answer_space"][-1])


def h_last_mentioned(item):
    """The last 'go to state X' target printed in the table (row 4, symbol c)."""
    last = None
    for r in item["rules"]:
        for t in re.findall(r"go to state (\d+)", r):
            last = t
    return last


# ---------------------------------------------------------------- 1-step
def h_first_step(item):
    """Apply only the FIRST symbol to the start state."""
    trans, seq, s0, _ = _parse(item)
    return _step(trans, s0, seq[0])


def h_last_step(item):
    """Apply only the LAST symbol to the start state."""
    trans, seq, s0, _ = _parse(item)
    return _step(trans, s0, seq[-1])


def h_majority_symbol_step(item):
    """Apply the most frequent symbol once to the start state."""
    trans, seq, s0, _ = _parse(item)
    maj = Counter(seq).most_common(1)[0][0]
    return _step(trans, s0, maj)


def h_row_mode(item):
    """Most common target in the start state's row (ties: first listed)."""
    trans, seq, s0, _ = _parse(item)
    targets = [trans[(s0, ch)] for ch in "abc" if (s0, ch) in trans]
    return Counter(targets).most_common(1)[0][0] if targets else None


def h_exclude_one_step(item):
    """Fixed-point-free symbols => for a 2-symbol string the answer can never be P_first(s0) or P_last(s0).
    Exclude those and pick the smallest remaining state (about 1/3 at level 1 instead of 1/5)."""
    trans, seq, s0, space = _parse(item)
    bad = {_step(trans, s0, seq[0]), _step(trans, s0, seq[-1])}
    rest = [a for a in space if a not in bad]
    return rest[0] if rest else None


def h_exclude_one_step_and_start(item):
    """As above but also exclude the start state ('surely it moved')."""
    trans, seq, s0, space = _parse(item)
    bad = {_step(trans, s0, seq[0]), _step(trans, s0, seq[-1]), s0}
    rest = [a for a in space if a not in bad]
    return rest[0] if rest else None


# ---------------------------------------------------------------- 2-step
def h_first_two_steps(item):
    """Apply the first two symbols only (this IS the full computation at level 1)."""
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
    """Apply the first symbol, then the last symbol."""
    trans, seq, s0, _ = _parse(item)
    s = _step(trans, s0, seq[0])
    return _step(trans, s, seq[-1]) if s is not None else None


def h_all_but_last(item):
    """STRUCTURAL-FLAVOURED but cheap to state: the state one step before the end can never be the answer
    (derangements), so 'stop one step short' is 0% by construction. Included to show the below-chance trap."""
    trans, seq, s0, _ = _parse(item)
    s = s0
    for ch in seq[:-1]:
        s = _step(trans, s, ch)
    return s


# ---------------------------------------------------------------- count-based (order-blind) closed forms
def _power(trans, s, ch, k):
    # reduce k modulo the cycle length through s (cycle length <= 5) so this is a lookup, not iteration
    cyc = [s]
    x = _step(trans, s, ch)
    while x is not None and x != s and len(cyc) <= 6:
        cyc.append(x)
        x = _step(trans, x, ch)
    return cyc[k % len(cyc)]


def h_majority_symbol_power(item):
    """Ignore minority symbols: answer = P_maj^(count_maj)(s0), reduced modulo the cycle length."""
    trans, seq, s0, _ = _parse(item)
    maj, k = Counter(seq).most_common(1)[0]
    return _power(trans, s0, maj, k)


def h_count_product(item):
    """Order-blind: apply P_a^(n_a), then P_b^(n_b), then P_c^(n_c). Exact iff the symbols commute."""
    trans, seq, s0, _ = _parse(item)
    c = Counter(seq)
    s = s0
    for ch in "abc":
        if c[ch]:
            s = _power(trans, s, ch, c[ch])
    return s


def heuristics():
    return {
        "start": h_start,
        "last_option": h_last_option,
        "last_mentioned_state": h_last_mentioned,
        "first_step": h_first_step,
        "last_step": h_last_step,
        "majority_symbol_step": h_majority_symbol_step,
        "row_mode": h_row_mode,
        "exclude_one_step_min": h_exclude_one_step,
        "exclude_one_step_and_start_min": h_exclude_one_step_and_start,
        "first_two_steps": h_first_two_steps,
        "last_two_from_start": h_last_two_from_start,
        "first_then_last": h_first_then_last,
        "majority_symbol_power": h_majority_symbol_power,
        "count_product_order_blind": h_count_product,
    }


# ---------------------------------------------------------------- structural checks (NOT shallow; report only)
def _compose(p, q):
    """Apply p then q."""
    return tuple(q[p[s]] for s in range(len(p)))


def composed_permutation(item):
    """STRUCTURAL: the full composed permutation g (answer = g[start])."""
    trans, seq, s0, space = _parse(item)
    n = len(space)
    g = tuple(range(n))
    for ch in seq:
        g = tuple(int(trans[(str(g[s]), ch)]) for s in range(n))
    return g


def structural_checks(item):
    """STRUCTURAL (uses full simulation): identity / fixed points / order of g, group order of <a,b,c>,
    reverse-order answer, stop-one-short answer, minimal equivalent word length.
    A model never sees these; use only for audits."""
    trans, seq, s0, space = _parse(item)
    n = len(space)
    gens = [tuple(int(trans[(str(s), ch)]) for s in range(n)) for ch in "abc"]
    g = composed_permutation(item)
    e = tuple(range(n))
    q, k = g, 1
    while q != e:
        q, k = _compose(q, g), k + 1
    dist = {e: 0}
    frontier = [e]
    while frontier:
        nf = []
        for x in frontier:
            for h in gens:
                y = _compose(x, h)
                if y not in dist:
                    dist[y] = dist[x] + 1
                    nf.append(y)
        frontier = nf
    s = s0
    for ch in reversed(seq):
        s = trans[(s, ch)]
    return {
        "g_is_identity": g == e,
        "g_fixed_points": sum(g[i] == i for i in range(n)),
        "g_order": k,
        "group_order": len(dist),
        "min_word_len": dist[g],
        "reverse_order_answer": s,
        "stop_one_short_answer": h_all_but_last(item),
        "single_symbol_string": len(set(seq)) == 1,
        "distinct_symbol_perms": len(set(gens)),
    }
