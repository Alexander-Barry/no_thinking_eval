"""
Shortcut audit: score cheap, shallow heuristics against the gold answers so that a strong model result on a
"deep" family can be checked against what needs no depth at all.

    python audit.py            # all families, all levels, prints a table per family x level

Generic baselines (every family):
  majority   the most common gold answer within (family, level)
  first      the first option listed in the item's answer space
Family-specific baselines live in heuristics/heur_<family>.py (one module per family, written during the
2026-09-06 shortcut hunt): position, 0-step and 1-step, closed-form and textual-cue heuristics. Every one is scored
per level and the best per family x level is folded into results/baselines.json.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

FILES = ["data/generated.jsonl"]


def load():
    items = []
    for p in FILES:
        for l in open(p, encoding="utf-8"):
            if l.strip():
                items.append(json.loads(l))
    return items


BEST: dict = defaultdict(dict)  # (family, level) -> best shallow-heuristic accuracy seen, written to results/baselines.json
BEST_NAME: dict = defaultdict(dict)  # (family, level) -> name of that heuristic


def record(fam, L, acc, name):
    if acc > BEST[fam].get(L, -1.0):
        BEST[fam][L], BEST_NAME[fam][L] = acc, name


def table(name, cells, record=True):
    """cells: {(family, level): (correct, n)}. record=True also folds the cell into the best-heuristic baseline;
    pass record=False for structural checks that are not heuristics (e.g. the circuit liveness fraction)."""
    for (f, L), (c, n) in cells.items():
        if n and record:
            globals()["record"](f, L, c / n, name.split(" (")[0].split(":")[0])
    fams = sorted({f for f, _ in cells})
    levels = sorted({L for _, L in cells})
    print(f"\n{name}")
    print(f"{'family':24s}" + "".join(f"{'L' + str(L):>8s}" for L in levels))
    for f in fams:
        row = ""
        for L in levels:
            c, n = cells.get((f, L), (None, 0))
            row += f"{(f'{c / n:.2f}' if n else ''):>8s}"
        print(f"{f:24s}{row}")


def generic(items):
    maj, first = {}, {}
    by = defaultdict(list)
    for it in items:
        by[(it["family"], it["difficulty"]["level"])].append(it)
    for key, its in by.items():
        top = Counter(it["answer"] for it in its).most_common(1)[0][1]
        maj[key] = (top, len(its))
        first[key] = (sum(it["answer"] == it["answer_space"][0] for it in its), len(its))
    table("majority-answer baseline (upper bound on 'always say the common answer')", maj)
    table("first-option baseline", first)


def plugin_heuristics(items):
    """Per-family heuristic modules in heuristics/heur_<family>.py (written during the 2026-09-06 shortcut hunt).
    Each exposes FAMILY and heuristics() -> {name: fn(item) -> predicted answer or None}. Every heuristic is scored
    per level and folded into the baseline; the table shows the best one per cell and names it."""
    import glob
    import importlib.util
    import os
    by = defaultdict(list)
    for it in items:
        by[(it["family"], it["difficulty"]["level"])].append(it)
    for path in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "heuristics", "heur_*.py"))):
        spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            fam, heur = mod.FAMILY, mod.heuristics()
        except Exception as e:  # a broken plugin must not silence the rest of the audit
            print(f"\n[plugin {os.path.basename(path)} failed to load: {e}]")
            continue
        levels = sorted(L for f, L in by if f == fam)
        if not levels:
            continue
        print(f"\n{fam}: plugin heuristics ({len(heur)}), accuracy per level; chance = 1/|answer space|")
        print(f"{'heuristic':44s}" + "".join(f"{'L' + str(L):>8s}" for L in levels))
        chance = "".join(f"{1 / len(by[(fam, L)][0]['answer_space']):>8.2f}" for L in levels)
        print(f"{'(chance)':44s}{chance}")
        best = {}
        for name, fn in heur.items():
            row = ""
            for L in levels:
                its = by[(fam, L)]
                c = n = 0
                for it in its:
                    try:
                        pred = fn(it)
                    except Exception:
                        pred = None
                    if pred is not None:
                        n += 1
                        c += pred == it["answer"]
                # a heuristic that abstains is scored as chance on the items it skips
                acc = (c + (len(its) - n) / len(its[0]["answer_space"])) / len(its) if its else 0.0
                row += f"{acc:>8.2f}"
                if acc > best.get(L, (0.0, ""))[0]:
                    best[L] = (acc, name)
                record(fam, L, acc, name)
            print(f"{name[:44]:44s}{row}")
        print(f"{'BEST':44s}" + "".join(f"{best[L][0]:>8.2f}" for L in levels))
        print("  best by level: " + "; ".join(f"L{L}: {best[L][1]}" for L in levels))


if __name__ == "__main__":
    items = load()
    generic(items)
    plugin_heuristics(items)
    out = {f: {str(L): {"acc": round(v, 4), "name": BEST_NAME[f][L]} for L, v in d.items()} for f, d in BEST.items()}
    json.dump(out, open("results/baselines.json", "w"), indent=1)
    print("\nwrote results/baselines.json (best shallow-heuristic accuracy per family x level)")
