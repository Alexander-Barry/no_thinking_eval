"""Shallow-heuristic hunt for the chain_lookup family (single-forward-pass eval suite).

Self-contained (json / re / math / collections only).  Imported by the project audit via heuristics();
run directly for the per-level table plus structural checks:

    python heur_chain_lookup.py [path/to/generated.jsonl]

Generator recap (generators.gen_chain_lookup): `hops` tables of 10 entries.  Table h maps a full permutation of
CHAIN_VOCAB[h-1] (keys, listed in rng.sample order) onto an independently shuffled copy of the next layer's words;
the last table maps onto a shuffled permutation of the ten digits.  Input = a uniformly chosen Table-1 key.
Each table is therefore a bijection between two disjoint word layers, so the composed map has no fixed points,
cycles, collisions or absorbing states -- the only cross-layer collision in CHAIN_VOCAB is the word "gold"
(colour layer 1 AND metal layer 7), which matters for hops >= 8 (levels 5-7).
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter

FAMILY = "chain_lookup"

# CHAIN_VOCAB[0] of the generator: a model that memorised the generator could use the canonical position of the
# input word (vocabulary-leak heuristics below).
VOCAB0 = ["dog", "cat", "cow", "pig", "hen", "fox", "owl", "bee", "ant", "eel"]

_TABLE = re.compile(r"^Table (\d+): (.*)\.$")


# --------------------------------------------------------------------------------------------------------------
# parsing helpers (prompt-visible information only)
# --------------------------------------------------------------------------------------------------------------
def parse(item):
    """-> list of tables in prompt order; each table is an ordered list of (key, value) pairs."""
    tables = []
    for r in item["rules"]:
        # v3.1 wording (signed off 2026-09-06): "g_1 is given by: g_1(dog) = red, ..."
        m = re.match(r"^g_(\d+) is given by: (.*)\.$", r.strip())
        if m:
            tables.append(re.findall(r"g_\d+\((\w+)\) = (\w+)", m.group(2)))
            continue
        m = _TABLE.match(r.strip())  # older wording "Table 1: dog -> red, ..."
        if m:
            tables.append([tuple(p.split(" -> ")) for p in m.group(2).split(", ")])
    return tables


def keys(t):
    return [k for k, _ in t]


def vals(t):
    return [v for _, v in t]


def lookup(t, w):
    for k, v in t:
        if k == w:
            return v
    return None


def row_of_key(t, w):
    ks = keys(t)
    return ks.index(w) if w in ks else None


def row_of_val(t, w):
    vs = vals(t)
    return vs.index(w) if w in vs else None


def digit_at_row(last, r):
    return vals(last)[r] if r is not None and 0 <= r < len(last) else None


def hop(item, T, n):
    """word reached after n forward lookups from the input (n <= len(T)); None if the chain breaks."""
    w = item["input"]
    for h in range(n):
        w = lookup(T[h], w)
        if w is None:
            return None
    return w


# --------------------------------------------------------------------------------------------------------------
# heuristics: item -> predicted digit (str in answer_space) or None
# --------------------------------------------------------------------------------------------------------------
def h_const_majority(it):
    """always the family-wide most common gold digit ('2')."""
    return "2"


def h_last_first(it):
    """first digit listed in the last table."""
    return vals(parse(it)[-1])[0]


def h_last_last(it):
    """last digit listed in the last table."""
    return vals(parse(it)[-1])[-1]


def h_last_middle(it):
    """middle (row 5) digit of the last table."""
    T = parse(it)
    return vals(T[-1])[len(T[-1]) // 2]


def h_row_input(it):
    """0-step row alignment: digit at the input's Table-1 row index, read in the last table."""
    T = parse(it)
    return digit_at_row(T[-1], row_of_key(T[0], it["input"]))


def h_row_input_mirror(it):
    """0-step row alignment, mirrored (row 9 - r)."""
    T = parse(it)
    r = row_of_key(T[0], it["input"])
    return digit_at_row(T[-1], len(T[-1]) - 1 - r) if r is not None else None


def h_row_hop1(it):
    """1 lookup, then row-align: row of the reached word among Table-2 KEYS -> same row in the last table.
    (Identical to the full solution when hops == 2, i.e. level 1.)"""
    T = parse(it)
    if len(T) < 2:
        return None
    return digit_at_row(T[-1], row_of_key(T[1], hop(it, T, 1)))


def h_row_hop1_valpos(it):
    """1 lookup, then row-align using the reached word's row among Table-1 VALUES (== input row; sanity twin of h_row_input)."""
    T = parse(it)
    return digit_at_row(T[-1], row_of_val(T[0], hop(it, T, 1)))


def h_row_hop2(it):
    """2 lookups, then row-align via Table-3 KEYS. (Identical to the full solution when hops == 3, level 2.)"""
    T = parse(it)
    if len(T) < 3:
        return None
    return digit_at_row(T[-1], row_of_key(T[2], hop(it, T, 2)))


def h_row_hop2_valpos(it):
    """2 lookups, then row-align via the reached word's row among Table-2 VALUES."""
    T = parse(it)
    if len(T) < 3:
        return None
    return digit_at_row(T[-1], row_of_val(T[1], hop(it, T, 2)))


def h_input_row_digit(it):
    """answer = the input's row index in Table 1, read as a numeral."""
    r = row_of_key(parse(it)[0], it["input"])
    return str(r) if r is not None else None


def h_hop1_row_digit(it):
    """answer = row index (in Table 2) of the 1-hop word, read as a numeral."""
    T = parse(it)
    r = row_of_key(T[1], hop(it, T, 1)) if len(T) > 1 else None
    return str(r) if r is not None else None


def h_alpha_rank_row(it):
    """alphabetical rank of the input among Table-1 keys -> digit at that row of the last table."""
    T = parse(it)
    return digit_at_row(T[-1], sorted(keys(T[0])).index(it["input"]))


def h_alpha_rank_digit(it):
    """alphabetical rank of the input among Table-1 keys, read as a numeral."""
    return str(sorted(keys(parse(it)[0])).index(it["input"]))


def h_vocab_index_row(it):
    """VOCAB LEAK: canonical CHAIN_VOCAB[0] index of the input -> digit at that row of the last table."""
    T = parse(it)
    return digit_at_row(T[-1], VOCAB0.index(it["input"])) if it["input"] in VOCAB0 else None


def h_vocab_index_digit(it):
    """VOCAB LEAK: canonical CHAIN_VOCAB[0] index of the input, read as a numeral."""
    return str(VOCAB0.index(it["input"])) if it["input"] in VOCAB0 else None


def h_prompt_digit_freq(it):
    """most frequent digit character anywhere in the rendered prompt (table labels included); ties -> smallest."""
    c = Counter(ch for ch in it["prompt"] if ch.isdigit())
    m = max(c.values())
    return min(d for d, n in c.items() if n == m)


def h_hops_mod10(it):
    """answer = number of tables mod 10."""
    return str(len(parse(it)) % 10)


def h_end_first_value(it):
    """one lookup from the END: first value listed in the penultimate table, looked up in the last table."""
    T = parse(it)
    return lookup(T[-1], vals(T[-2])[0]) if len(T) >= 2 else None


def h_end_last_value(it):
    """one lookup from the END: last value listed in the penultimate table, looked up in the last table."""
    T = parse(it)
    return lookup(T[-1], vals(T[-2])[-1]) if len(T) >= 2 else None


def h_end_row_input(it):
    """row-align then one lookup from the end: value at the input's row in the penultimate table -> last table.
    (Identical to the full solution when hops == 2.)"""
    T = parse(it)
    r = row_of_key(T[0], it["input"])
    return lookup(T[-1], vals(T[-2])[r]) if len(T) >= 2 and r is not None else None


def h_last_alpha_first_key(it):
    """digit of the alphabetically first key in the last table (would matter if keys were ever sorted)."""
    T = parse(it)
    return lookup(T[-1], min(keys(T[-1])))


def h_last_alpha_last_key(it):
    """digit of the alphabetically last key in the last table."""
    T = parse(it)
    return lookup(T[-1], max(keys(T[-1])))


def h_nearest_digit_to_hop1_key(it):
    """attention-proximity: the digit character textually nearest to the 1-hop word's occurrence as a KEY
    (correct when hops == 2; otherwise mostly picks up a 'Table N:' label)."""
    T = parse(it)
    w = hop(it, T, 1)
    if w is None or len(T) < 2:
        return None
    p = it["prompt"]
    pos = p.find(w + " -> ", p.find("Table 2:"))
    if pos < 0:
        return None
    best, bd = None, 10 ** 9
    for i, ch in enumerate(p):
        if ch.isdigit() and abs(i - pos) < bd:
            best, bd = ch, abs(i - pos)
    return best


def heuristics():
    return {
        "const_majority_2": h_const_majority,
        "last_table_first_digit": h_last_first,
        "last_table_last_digit": h_last_last,
        "last_table_middle_digit": h_last_middle,
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
        "vocab_index_row_align": h_vocab_index_row,
        "vocab_index_as_digit": h_vocab_index_digit,
        "prompt_digit_frequency": h_prompt_digit_freq,
        "hops_mod_10": h_hops_mod10,
        "end_first_value_1hop": h_end_first_value,
        "end_last_value_1hop": h_end_last_value,
        "end_row_align_1hop": h_end_row_input,
        "last_alpha_first_key": h_last_alpha_first_key,
        "last_alpha_last_key": h_last_alpha_last_key,
        "nearest_digit_to_hop1_key": h_nearest_digit_to_hop1_key,
    }


# --------------------------------------------------------------------------------------------------------------
# scoring + structural checks (run directly)
# --------------------------------------------------------------------------------------------------------------
def _load(path):
    out = []
    for l in open(path, encoding="utf-8"):
        if l.strip():
            it = json.loads(l)
            if it["family"] == FAMILY:
                out.append(it)
    return out


def _pearson(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (sx * sy) if sx and sy else 0.0


def _score(items):
    levels = sorted({it["difficulty"]["level"] for it in items})
    by = {L: [it for it in items if it["difficulty"]["level"] == L] for L in levels}
    hops = {L: len(parse(by[L][0])) for L in levels}
    print(f"{'heuristic':30s}" + "".join(f"{'L' + str(L):>7s}" for L in levels) + "   flagged (>= chance+0.10)")
    print(f"{'(hops)':30s}" + "".join(f"{hops[L]:>7d}" for L in levels))
    print(f"{'(n)':30s}" + "".join(f"{len(by[L]):>7d}" for L in levels))
    print(f"{'chance':30s}" + "".join(f"{1 / len(by[L][0]['answer_space']):>7.2f}" for L in levels))
    for name, fn in heuristics().items():
        accs, flags = [], []
        for L in levels:
            c = sum(fn(it) == it["answer"] for it in by[L])
            a = c / len(by[L])
            accs.append(a)
            if a >= 1 / len(by[L][0]["answer_space"]) + 0.10:
                flags.append(f"L{L}")
        print(f"{name:30s}" + "".join(f"{a:>7.2f}" for a in accs) + ("   " + ",".join(flags) if flags else ""))


def _structural(items):
    print("\n=== structural checks (labelled; rationale used only where stated) ===")
    levels = sorted({it["difficulty"]["level"] for it in items})
    by = {L: [it for it in items if it["difficulty"]["level"] == L] for L in levels}

    # 1. gold digit distribution
    print("\n[1] gold digit counts per level (uniform expected; chi2 critical 16.9 at p=.05, df=9)")
    for L in levels:
        c = Counter(it["answer"] for it in by[L])
        n = len(by[L])
        chi = sum((c.get(d, 0) - n / 10) ** 2 / (n / 10) for d in "0123456789")
        print(f"  L{L}: " + " ".join(f"{d}:{c.get(d, 0):>2d}" for d in "0123456789") + f"   chi2={chi:.1f}")
    c = Counter(it["answer"] for it in items)
    n = len(items)
    chi = sum((c.get(d, 0) - n / 10) ** 2 / (n / 10) for d in "0123456789")
    print("  all: " + " ".join(f"{d}:{c.get(d, 0):>2d}" for d in "0123456789") + f"   chi2={chi:.1f}")

    # 2. answer row in last table vs input row
    print("\n[2] row of the answer's key in the last table vs the input's row in Table 1")
    xs, ys, same = [], [], 0
    for it in items:
        T = parse(it)
        xs.append(row_of_key(T[0], it["input"]))
        ys.append(row_of_val(T[-1], it["answer"]))
        same += xs[-1] == ys[-1]
    print(f"  answer-row histogram: {sorted(Counter(ys).items())}")
    print(f"  input-row histogram:  {sorted(Counter(xs).items())}")
    print(f"  P(same row) = {same / len(items):.3f} (chance 0.10); pearson r = {_pearson(xs, ys):+.3f}")

    # 3. row preservation between adjacent tables (all entries, not just the path)
    print("\n[3] row preservation: P(row of word as VALUE in table k == row of that word as KEY in table k+1), all entries")
    hits = tot = 0
    for it in items:
        T = parse(it)
        for k in range(len(T) - 1):
            for r, (_, v) in enumerate(T[k]):
                tot += 1
                hits += row_of_key(T[k + 1], v) == r
    print(f"  overall {hits}/{tot} = {hits / tot:.3f} (chance 0.10)")

    # 4. sortedness of keys / digits
    print("\n[4] sortedness")
    kt = ktot = 0
    dsorted = 0
    rank_corr = []
    rows, digs = [], []
    for it in items:
        T = parse(it)
        for t in T:
            ktot += 1
            kt += keys(t) == sorted(keys(t))
        d = vals(T[-1])
        dsorted += d == sorted(d) or d == sorted(d, reverse=True)
        ks = keys(T[-1])
        ranks = [sorted(ks).index(k) for k in ks]
        rank_corr.append(_pearson(ranks, [int(x) for x in d]))
        for r, v in enumerate(d):
            rows.append(r)
            digs.append(int(v))
    print(f"  tables with alphabetically sorted keys: {kt}/{ktot}")
    print(f"  last tables with sorted digits (asc or desc): {dsorted}/{len(items)}")
    print(f"  mean corr(alpha rank of key, digit) in last table: {sum(rank_corr) / len(rank_corr):+.3f}")
    print(f"  corr(row index, digit) in last table over all entries: {_pearson(rows, digs):+.3f}")

    # 5. duplicate words within a prompt, 'gold' on the path (RATIONALE used here)
    print("\n[5] cross-layer word collisions (prompt) and paths through them (RATIONALE-based)")
    dup_items = Counter()
    gold_twice = gold_l1 = gold_l7 = 0
    occ_ok = {"first": Counter(), "last": Counter()}

    def _string_matcher(it, T, which):
        """full-depth reader that resolves 'w ->' by its FIRST / LAST textual occurrence (wrong table when w is ambiguous)."""
        w = it["input"]
        for _ in range(len(T)):
            cands = [lookup(t, w) for t in T if lookup(t, w) is not None]
            if not cands:
                break
            w = cands[0] if which == "first" else cands[-1]
            if w.isdigit():
                break
        return w == it["answer"]

    for it in items:
        T = parse(it)
        allk = Counter(k for t in T for k in keys(t))
        dups = tuple(sorted(w for w, n in allk.items() if n > 1))
        if dups:
            dup_items[dups] += 1
        path = it["rationale"].split(" -> ")
        gold_twice += path.count("gold") == 2
        gold_l1 += len(path) > 1 and path[1] == "gold"
        gold_l7 += len(path) > 7 and path[7] == "gold"
        for which in occ_ok:
            occ_ok[which][it["difficulty"]["level"]] += _string_matcher(it, T, which)
    print(f"  items whose prompt has a key appearing in two tables: {sum(dup_items.values())}/{len(items)}  {dict(dup_items)}")
    print(f"  paths through 'gold' at layer 1: {gold_l1}; at layer 7: {gold_l7}; both: {gold_twice}")
    for which in occ_ok:
        tot = sum(occ_ok[which].values())
        print(f"  'resolve w -> by its {which.upper()} occurrence' accuracy: {tot}/{len(items)} = {tot / len(items):.3f}"
              "  per level: " + " ".join(f"L{L}:{occ_ok[which][L]}/{len(by[L])}" for L in levels))
    print("  (both are full-depth readers; the deficit at hops >= 8 is the 'gold' ambiguity, i.e. noise, not a shortcut)")

    # 6. depth: bijections, disjoint layers, distinct path words
    print("\n[6] depth check: every table a bijection, layers disjoint, path words distinct (RATIONALE for the path)")
    bij = disj = distinct = 0
    for it in items:
        T = parse(it)
        bij += all(len(set(keys(t))) == len(t) == len(set(vals(t))) for t in T) and all(
            set(vals(T[k])) == set(keys(T[k + 1])) for k in range(len(T) - 1))
        layers = [set(keys(t)) for t in T] + [set(vals(T[-1]))]
        disj += all(not (layers[i] & layers[j]) for i in range(len(layers)) for j in range(i + 1, len(layers)))
        path = it["rationale"].split(" -> ")
        distinct += len(set(path)) == len(path)
    print(f"  bijection chain: {bij}/{len(items)}; all layers pairwise disjoint: {disj}/{len(items)}; "
          f"path words all distinct: {distinct}/{len(items)}")
    print("  => composed map is a random bijection words->digits with no fixed points / cycles / absorption;"
          " hops = sequential lookups needed\n     (caveat: table composition is associative, so a parallel-prefix"
          " reader needs only ~log2(hops) serial stages given width to compose whole tables)")

    # 7. cross-item repetition
    print("\n[7] cross-item repetition (same Table-1 line or same last-table line reused)")
    t1 = Counter(it["rules"][0] for it in items)
    tl = Counter([r for r in it["rules"] if r.startswith("Table")][-1] for it in items)
    print(f"  duplicated Table-1 lines: {sum(n - 1 for n in t1.values() if n > 1)}; "
          f"duplicated last-table lines: {sum(n - 1 for n in tl.values() if n > 1)}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/generated.jsonl"
    items = _load(path)
    print(f"{FAMILY}: {len(items)} items from {path}\n")
    _score(items)
    _structural(items)
