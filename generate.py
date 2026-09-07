"""
Generates the items for the single-forward-pass rule tasks.

    python generate.py --seed 0        writes 40 items per (family, level) cell to data/generated.jsonl

There are four families. Each is a small system whose rules are written out in full in the prompt, and each item
asks for the result of running the system on a one-token input for a set number of steps: a permutation of the
letters applied k times, a chain of k word-to-word maps, a word of k permutations run on a five-state machine, and
a two-term recurrence over digits run for k terms. Every step is a bijection, so no step loses information and k
steps never collapse into fewer.

The depth grid is shared by all families, so a level means the same number of steps everywhere. Within each
(family, level) cell the answers are balanced by construction: the wanted answers are laid out first, and each
input is found by running the system backwards from its answer.

The prompt texts in this file, the header, the trailers and each family's rule wording, were signed off word for
word by the author. Don't change them without going back to them.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"

HEADER = (
    "Hello Claude, today I'm looking at how capable LLMs are of reasoning in a single forward pass, "
    "without generating any tokens first. For this I'll ask you to respond to a question without thinking "
    "at all, just by writing your answer out straight away. Try your best, even if it feels like just guessing. "
    "The questions are built so that every possible answer is equally likely before you read the input, so there "
    "is nothing to gain from favouring a common answer.\n\n"
    "Each question is a set of rules that maps a single-token input to a single-token output. First I'll give you "
    "the rules, and then at the end of the prompt I'll give you the input token. Sometimes there will be a stretch "
    "of filler symbols after the input (or before it); they carry no information and are only there to give you "
    "extra positions, so please ignore them and answer as before. Please never write out working or intermediate "
    "steps, even if your answer is a guess. Thanks for helping with this."
)
# {phrase} is filled by answer_phrase. At run time the poem condition swaps TRAILER for POEM_TRAILER, so even then
# nothing but filler follows the input.
TRAILER = "Please answer with just {phrase}, nothing else, straight away, and without writing out any intermediate steps."
POEM_TRAILER = ("Before you answer, please write a short poem (around 100 words) about a dog. "
                "Then on a new line give just the answer as {phrase}, nothing else.")


def answer_phrase(answer_space) -> str:
    """'a single digit from 0-4' or 'a single capital letter from A-Z'."""
    sp = sorted(answer_space)
    if all(a.isdigit() for a in sp):
        return f"a single digit from {sp[0]}-{sp[-1]}"
    if all(a.isalpha() and a.isupper() for a in sp):
        return f"a single capital letter from {sp[0]}-{sp[-1]}"
    raise ValueError(f"unrecognised answer space: {sp}")


def render(rules: list[str], input_text: str, answer_space) -> str:
    body = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rules))
    return f"{HEADER}\n\nRules:\n{body}\n\n{TRAILER.format(phrase=answer_phrase(answer_space))}\n\nInput: {input_text}"


def item(*, id_, family, rules, input_text, answer, answer_space, rationale):
    assert answer in answer_space, (answer, answer_space)
    return {
        "id": id_,
        "family": family,
        "prompt": render(rules, input_text, answer_space),
        "rules": rules,
        "input": input_text,
        "answer": answer,
        "answer_space": list(answer_space),
        "rationale": rationale,
    }


# --------------------------------------------------------------------------
# chain_lookup: k word -> word maps over disjoint vocabularies, the last one to digits
# --------------------------------------------------------------------------
CHAIN_VOCAB = [
    ["dog", "cat", "cow", "pig", "hen", "fox", "owl", "bee", "ant", "eel"],
    ["red", "blue", "green", "black", "white", "pink", "gray", "brown", "olive", "cyan"],
    ["circle", "square", "star", "cross", "arrow", "heart", "moon", "wave", "ring", "leaf"],
    ["north", "south", "east", "west", "up", "down", "left", "right", "in", "out"],
    ["apple", "pear", "plum", "fig", "lime", "kiwi", "date", "peach", "grape", "mango"],
    ["hammer", "saw", "drill", "wrench", "chisel", "pliers", "file", "rake", "hoe", "axe"],
    ["oak", "elm", "ash", "fir", "pine", "yew", "birch", "cedar", "maple", "palm"],
    ["iron", "gold", "zinc", "tin", "lead", "copper", "silver", "nickel", "cobalt", "steel"],
    ["sofa", "chair", "desk", "lamp", "bed", "stool", "shelf", "bench", "crib", "couch"],
    ["violin", "drum", "flute", "harp", "piano", "cello", "horn", "banjo", "oboe", "tuba"],
    ["rain", "snow", "hail", "fog", "wind", "storm", "frost", "mist", "sleet", "thunder"],
    ["france", "spain", "italy", "japan", "chile", "peru", "kenya", "egypt", "india", "cuba"],
    ["shirt", "hat", "coat", "boot", "sock", "glove", "scarf", "belt", "tie", "vest"],
    ["bread", "rice", "cheese", "soup", "cake", "pasta", "salad", "pie", "stew", "toast"],
    ["ruby", "opal", "jade", "onyx", "pearl", "topaz", "amber", "coral", "ivory", "agate"],
    ["tulip", "rose", "lily", "daisy", "iris", "poppy", "lotus", "orchid", "peony", "violet"],
    # layers 17-32, added 2026-09-07 for depths 20, 24 and 32
    ["car", "bus", "train", "bike", "boat", "truck", "plane", "ship", "tram", "van"],
    ["hand", "foot", "knee", "elbow", "nose", "ear", "chin", "wrist", "thumb", "heel"],
    ["baker", "nurse", "judge", "pilot", "farmer", "tailor", "miner", "clerk", "monk", "poet"],
    ["golf", "tennis", "rugby", "chess", "boxing", "rowing", "polo", "judo", "hockey", "cricket"],
    ["kitchen", "attic", "cellar", "garage", "chapel", "tower", "barn", "hut", "palace", "cabin"],
    ["mars", "venus", "jupiter", "saturn", "neptune", "pluto", "comet", "nebula", "orbit", "eclipse"],
    ["silk", "wool", "linen", "cotton", "denim", "velvet", "satin", "tweed", "nylon", "lace"],
    ["tea", "coffee", "milk", "juice", "cider", "cocoa", "beer", "wine", "soda", "water"],
    ["robin", "eagle", "crow", "swan", "heron", "finch", "pigeon", "parrot", "sparrow", "falcon"],
    ["salmon", "trout", "cod", "tuna", "shark", "carp", "pike", "perch", "herring", "haddock"],
    ["waltz", "tango", "salsa", "polka", "jazz", "blues", "opera", "ballet", "samba", "disco"],
    ["hill", "valley", "cliff", "canyon", "desert", "glacier", "island", "marsh", "meadow", "plateau"],
    ["spoon", "fork", "knife", "plate", "bowl", "cup", "kettle", "pan", "jug", "ladle"],
    ["puzzle", "kite", "doll", "marble", "yoyo", "domino", "dice", "card", "robot", "teddy"],
    ["algebra", "history", "physics", "biology", "chemistry", "geology", "grammar", "drama", "ethics", "logic"],
    ["beetle", "wasp", "moth", "hornet", "gnat", "termite", "locust", "mantis", "aphid", "flea"],
]
# The layers must be disjoint. A word in two layers would appear twice as a key, and a reader that resolves it by
# position would take a wrong hop for a reason unrelated to depth.
assert len({w for v in CHAIN_VOCAB for w in v}) == sum(len(v) for v in CHAIN_VOCAB), "CHAIN_VOCAB layers overlap"


def gen_chain_lookup(rng: random.Random, id_: str, hops: int, want_answer: str):
    assert 1 <= hops <= len(CHAIN_VOCAB)
    layers = [rng.sample(v, 10) for v in CHAIN_VOCAB[:hops]]
    digits = rng.sample(DIGITS, 10)
    layers.append(digits)
    if hops == 1:
        rules = ["This question concerns a single map, g_1, which sends every element of its domain to a digit from 0-9."]
    else:
        names = ", ".join(f"g_{h}" for h in range(1, hops)) + f" and g_{hops}"
        rules = [f"This question concerns a chain of {hops} maps, {names}. Each map sends every element of its domain to an "
                 "element of the next map's domain; the last map sends its elements to digits from 0-9."]
    maps = []
    for h in range(hops):
        src, dst = layers[h], layers[h + 1][:]
        rng.shuffle(dst)
        m = dict(zip(src, dst))
        maps.append(m)
        rules.append(f"g_{h + 1} is given by: " + ", ".join(f"g_{h + 1}({k}) = {v}" for k, v in m.items()) + ".")
    rules.append("Let x_0 be the input word, and for every n >= 1 let x_n = g_n(x_(n-1)).")
    rules.append(f"Answer with x_{hops}.")
    x = want_answer  # every map is a bijection: walk back from the wanted digit to the unique input word
    for m in reversed(maps):
        x = {v: k for k, v in m.items()}[x]
    path = [x]
    for m in maps:
        x = m[x]
        path.append(x)
    return item(id_=id_, family="chain_lookup", rules=rules, input_text=path[0], answer=path[-1],
                answer_space=sorted(digits), rationale=" -> ".join(path))


# --------------------------------------------------------------------------
# state_machine: a word of k permutations of 5 states applied to the input state
# --------------------------------------------------------------------------
N_STATES = 5
ALPHABET = "abcdef"
NUMBER_WORD = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


def gen_state_machine(rng: random.Random, id_: str, seq_len: int, want_answer: str):
    """Each symbol is a uniformly random permutation of the states. Fixed-point-free symbols would rule out the
    one-step states and push partial simulation away from chance, so they are not used. The word never repeats a
    symbol twice in a row, because a permutation returns to the same state after two identical steps two times in
    five, which let a simulator that stopped two symbols early beat chance. A symbol equal to the identity, two
    identical symbols, or a whole word equal to the identity are redrawn."""
    states = list(range(N_STATES))
    for _ in range(1000):
        trans, perms = {}, []
        for ch in ALPHABET:
            perm = states[:]
            rng.shuffle(perm)
            perms.append(tuple(perm))
            for s in states:
                trans[(s, ch)] = perm[s]
        if len(set(perms)) < len(ALPHABET) or tuple(states) in perms:
            continue
        seq: list[str] = []
        for _ in range(seq_len):
            ch = rng.choice(ALPHABET)
            while seq and ch == seq[-1]:
                ch = rng.choice(ALPHABET)
            seq.append(ch)
        g = list(states)  # g[s] = state reached from s after the whole word
        for ch in seq:
            g = [trans[(s, ch)] for s in g]
        if g == states:
            continue
        break
    target = int(want_answer)
    s0 = g.index(target)
    names = ", ".join(ALPHABET[:-1]) + f" and {ALPHABET[-1]}"
    rules = [f"This question concerns a set of {N_STATES} states, numbered 0-{N_STATES - 1}, and {NUMBER_WORD[len(ALPHABET)]} "
             f"permutations of those states, named {names}."]
    for ch in ALPHABET:
        rules.append(f"{ch} is given by: " + ", ".join(f"{ch}({s}) = {trans[(s, ch)]}" for s in states) + ".")
    rules.append("When I write a b c (x), I mean a(b(c(x))): the rightmost permutation is applied first.")
    # the word is written in composition order, i.e. reversed relative to the order of application
    rules.append("Let x_0 be the input state. Answer with " + " ".join(reversed(seq)) + " (x_0).")
    s, path = s0, [s0]
    for ch in seq:
        s = trans[(s, ch)]
        path.append(s)
    assert s == target
    return item(id_=id_, family="state_machine", rules=rules, input_text=str(s0), answer=str(s),
                answer_space=[str(t) for t in states], rationale="states visited: " + " ".join(map(str, path)))


# --------------------------------------------------------------------------
# sequence_mod: a two-term recurrence through a stated permutation f, k terms after the input
# --------------------------------------------------------------------------
def gen_sequence_mod(rng: random.Random, id_: str, steps: int, want_answer: str):
    """term n = (f(term n-1) + term n-2) mod 10 with a random permutation f. Going through f rather than adding the
    terms directly means there is no closed form to read off, which Fibonacci mod 10 has. The step on pairs,
    (a, b) -> (b, f(b) + a), is a bijection, so orbits never merge. An orbit that returns to its starting pair
    before the answer term is redrawn."""
    for _ in range(500):
        T = list(range(10))
        rng.shuffle(T)
        a0 = rng.randint(0, 9)
        cands = []
        for x0 in range(10):
            a, b, path, seen, ok = a0, x0, [x0], {(a0, x0)}, True
            for _ in range(steps):
                a, b = b, (T[b] + a) % 10
                if (a, b) in seen:
                    ok = False
                    break
                seen.add((a, b))
                path.append(b)
            if ok and str(path[-1]) == want_answer:
                cands.append((x0, path))
        if cands:
            x0, path = rng.choice(cands)
            break
    else:
        raise RuntimeError(id_)
    rules = ['This question concerns a sequence whose elements are digits from 0-9. We write "term 1" for the first '
             'element, "term 2" for the second, and so on.',
             f"Term 1 is {a0}. Term 2 is the input digit.",
             "Let f be the following permutation of the digits 0-9: " + ", ".join(f"f({d}) = {T[d]}" for d in range(10)) + ".",
             "For every n >= 3, term n = (f(term (n-1)) + term (n-2)) mod 10.",
             f"Answer with term {steps + 2}."]
    return item(id_=id_, family="sequence_mod", rules=rules, input_text=str(x0), answer=str(path[-1]),
                answer_space=list(DIGITS), rationale=f"terms 1..{steps + 2}: {a0} " + " ".join(map(str, path)))


# --------------------------------------------------------------------------
# iterate_map: one permutation of the 26 letters applied k times
# --------------------------------------------------------------------------
def gen_iterate_map(rng: random.Random, id_: str, steps: int, want_answer: str):
    """The permutation is a single 26-cycle, so applying g k times is k genuine steps for every k up to 13. On a
    shorter cycle of length n, g^k equals g^-(n-k), and a deep item becomes a few backward lookups. Cycles with
    three or more entries of the form g(L) = L+1 or L-1 are redrawn, so a Caesar-shift guess has nothing to work
    with."""
    assert 1 <= steps <= 13
    n = len(LETTERS)
    for _ in range(200):
        cyc = list(LETTERS)
        rng.shuffle(cyc)
        f = {cyc[i]: cyc[(i + 1) % n] for i in range(n)}
        if sum(abs(LETTERS.index(f[c]) - LETTERS.index(c)) == 1 for c in LETTERS) < 3:
            break
    inv = {v: k for k, v in f.items()}
    x0 = want_answer
    for _ in range(steps):
        x0 = inv[x0]
    x, path = x0, [x0]
    for _ in range(steps):
        x = f[x]
        path.append(x)
    rules = ["This question concerns a permutation g of the capital letters A-Z, given by: "
             + ", ".join(f"g({c}) = {f[c]}" for c in LETTERS) + ".",
             f"Apply g to the input letter exactly {steps} time{'s' if steps > 1 else ''} in succession.",
             "Answer with the letter you end on."]
    return item(id_=id_, family="iterate_map", rules=rules, input_text=x0, answer=x,
                answer_space=list(LETTERS), rationale="orbit: " + " -> ".join(path))


# --------------------------------------------------------------------------
# Ladder: level k = the k-th depth on the shared grid
# --------------------------------------------------------------------------
DEPTHS = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32]

FAMILY_DEPTHS = {
    "sequence_mod": [1, 2, 3, 4, 6, 8, 12, 16, 20],
    "iterate_map": [1, 2, 3, 4, 6, 8, 12],                  # a 26-cycle is honest only up to depth 13
    "state_machine": [1, 2, 3, 4, 6, 8, 12, 16, 20],
    "chain_lookup": [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32],  # one vocabulary layer per hop
}

GENERATORS = {"sequence_mod": gen_sequence_mod, "iterate_map": gen_iterate_map,
              "state_machine": gen_state_machine, "chain_lookup": gen_chain_lookup}

ANSWER_SPACE = {"sequence_mod": list(DIGITS), "iterate_map": list(LETTERS),
                "state_machine": [str(i) for i in range(N_STATES)], "chain_lookup": list(DIGITS)}


def stratified_targets(rng: random.Random, space: list[str], n: int) -> list[str]:
    """Each answer as equally often as n allows (n = 40 over 26 letters -> 14 letters twice, 12 once), shuffled."""
    pool = list(space) * (-(-n // len(space)))
    rng.shuffle(pool)
    return pool[:n]


def generate_all(seed: int, per_level: int) -> list[dict]:
    out = []
    for fam, depths in FAMILY_DEPTHS.items():
        for depth in depths:
            level = DEPTHS.index(depth) + 1
            targets = stratified_targets(random.Random(f"{seed}-{fam}-{level}-targets"), ANSWER_SPACE[fam], per_level)
            for i in range(per_level):
                rng = random.Random(f"{seed}-{fam}-{level}-{i}")
                it = GENERATORS[fam](rng, f"{fam}-L{level}-s{seed}-{i}", depth, targets[i])
                it["difficulty"] = {"level": level, "depth": depth}
                out.append(it)
    return out


def validate(items: list[dict], schema_path: str = "schema.json") -> None:
    schema = json.load(open(schema_path, encoding="utf-8"))
    required = set(schema["required"])
    families = set(schema["properties"]["family"]["enum"])
    answer = re.compile(schema["properties"]["answer"]["pattern"])
    ids = set()
    for it in items:
        assert required <= set(it), (it["id"], required - set(it))
        assert it["family"] in families, it["family"]
        assert answer.match(it["answer"]) and it["answer"] in it["answer_space"], it["id"]
        assert it["id"] not in ids, it["id"]
        ids.add(it["id"])
        assert it["prompt"].endswith(f"\n\nInput: {it['input']}"), it["id"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/generated.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-level", type=int, default=40)
    a = ap.parse_args()
    items = generate_all(a.seed, a.per_level)
    validate(items)
    with open(a.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} items to {a.out}")
    for fam, n in Counter(it["family"] for it in items).items():
        print(f"  {fam}: {n}")
