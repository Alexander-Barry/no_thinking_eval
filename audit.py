"""
Scores shallow heuristics against the gold answers, so that a model's accuracy on a cell can be compared with what
needs no depth at all.

    python audit.py            prints a table per family and writes results/baselines.json

Two generic baselines apply to every family: the most common answer in the cell, and the first option of the answer
space. The rest live in heuristics/heur_<family>.py, one module per family. Each exposes FAMILY and heuristics(),
a dict from a name to a function that takes an item and returns a predicted answer, or None to abstain, which is
scored as chance. The best accuracy in each cell and the heuristic that achieved it are written to
results/baselines.json.

The point is to catch a generator that leaks. After changing one, run the audit and check that every heuristic sits
at chance on the cells deeper than what it computes: a one-step lookup should solve depth 1 and nothing else.
"""
from __future__ import annotations

import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "generated.jsonl"
OUT = HERE / "results" / "baselines.json"


def load_cells() -> dict[str, dict[int, list[dict]]]:
    """family -> level -> items"""
    cells: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for line in DATA.read_text(encoding="utf-8").splitlines():
        if line.strip():
            it = json.loads(line)
            cells[it["family"]][it["difficulty"]["level"]].append(it)
    return cells


def plugin_heuristics() -> dict[str, dict]:
    """family -> {name: fn} from heuristics/heur_*.py"""
    out = {}
    for path in sorted((HERE / "heuristics").glob("heur_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out[mod.FAMILY] = mod.heuristics()
    return out


def cell_heuristics(items: list[dict], plugins: dict) -> dict:
    majority = Counter(it["answer"] for it in items).most_common(1)[0][0]
    return {"majority answer": lambda it: majority, "first option": lambda it: it["answer_space"][0], **plugins}


def score(fn, items: list[dict]) -> float:
    """Accuracy of a heuristic on a cell; items it abstains on (None or an exception) count as chance."""
    hits = abstain = 0
    for it in items:
        try:
            pred = fn(it)
        except Exception:
            pred = None
        if pred is None:
            abstain += 1
        else:
            hits += pred == it["answer"]
    return (hits + abstain / len(items[0]["answer_space"])) / len(items)


def main() -> None:
    cells = load_cells()
    plugins = plugin_heuristics()
    best: dict[str, dict[int, tuple[float, str]]] = defaultdict(dict)
    for fam, by_level in cells.items():
        levels = sorted(by_level)
        acc = {L: {name: score(fn, by_level[L]) for name, fn in cell_heuristics(by_level[L], plugins.get(fam, {})).items()}
               for L in levels}
        names = list(acc[levels[0]])
        for L in levels:
            best[fam][L] = max(((acc[L][n], n) for n in names), key=lambda t: t[0])
        print(f"\n{fam}: {len(names)} heuristics, accuracy per level")
        print(f"{'':44s}" + "".join(f"{'L' + str(L):>8s}" for L in levels))
        print(f"{'(chance)':44s}" + "".join(f"{1 / len(by_level[L][0]['answer_space']):>8.2f}" for L in levels))
        for name in names:
            print(f"{name[:44]:44s}" + "".join(f"{acc[L][name]:>8.2f}" for L in levels))
        print(f"{'BEST':44s}" + "".join(f"{best[fam][L][0]:>8.2f}" for L in levels))
        print("  best by level: " + "; ".join(f"L{L}: {best[fam][L][1]}" for L in levels))
    OUT.parent.mkdir(exist_ok=True)
    out = {fam: {str(L): {"acc": round(a, 4), "name": name} for L, (a, name) in sorted(d.items())} for fam, d in best.items()}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(HERE)} (best shallow-heuristic accuracy per family x level)")


if __name__ == "__main__":
    main()
