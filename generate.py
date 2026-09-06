"""
Generators for the single-forward-pass rule tasks (four families of bijective in-context maps).

    python generate.py --out data/generated.jsonl --seed 0            # 40 items per (family, depth) cell

Every prompt text here (header, trailers, poem trailer, family rule wording) was signed off by the author; do not edit
it without going back to them. A level is a depth on the grid shared by all families (DEPTHS); gold answers are
stratified per cell so every answer is equally frequent, and the input is solved backwards from the wanted answer.
"""
from __future__ import annotations

import argparse
import json
import random
import re

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# The prompt explains the experiment to the model and asks for its cooperation. A bare
# "Reply with ONLY... No explanation." instruction was tried first: Sonnet 5 and Opus 5 ignored it
# on a third to a half of items and started writing their working, and it isn't a nice way to ask.
# This framing got full compliance from every model tested.
PROMPT = {
    "header": (
        # Signed off verbatim by the user on 2026-09-06 (two paragraphs). Do not edit without explicit approval.
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
    ),
    # The trailer names the item's answer space; signed off by the user on 2026-09-06. {phrase} is filled by answer_phrase().
    "trailer": "Please answer with just {phrase}, nothing else, straight away, and without writing out any intermediate steps.",
    # Poem condition (generated filler): replaces the trailer, so the instruction sits before the Input line. Signed off 2026-09-06.
    "poem_trailer": "Before you answer, please write a short poem (around 100 words) about a dog. Then on a new line give just the answer as {phrase}, nothing else.",
}


def answer_phrase(answer_space) -> str:
    """'a single digit from 0-4' or 'a single capital letter from A-Z'."""
    sp = sorted(answer_space)
    if set(sp) <= {"Y", "N"}:
        return "the letter Y (for yes) or N (for no)"
    if all(a.isdigit() for a in sp):
        return f"a single digit from {sp[0]}-{sp[-1]}"
    if all(a.isalpha() and a.isupper() for a in sp):
        return f"a single capital letter from {sp[0]}-{sp[-1]}"
    raise ValueError(f"unrecognised answer space: {sp}")


_TRAILER_RE = re.compile(r"Please answer with just (.+?), nothing else, straight away, and without writing out any intermediate steps\.")


def with_poem(prompt: str) -> str:
    """The same prompt with the answer trailer replaced by the poem trailer (same answer phrase)."""
    m = _TRAILER_RE.search(prompt)
    if not m:
        raise ValueError("prompt has no recognised answer trailer")
    return prompt[:m.start()] + PROMPT["poem_trailer"].format(phrase=m.group(1)) + prompt[m.end():]


# The header greets the model by name. Data files are rendered once with the default; a runner for another
# provider swaps the name at send time with readdress(), so every model reads the same prompt apart from its own name.
ADDRESSEE = "Claude"


ADDRESSEE_BY_PREFIX = [("claude", "Claude"), ("gemini", "Gemini"), ("gemma", "Gemini"),
                       ("gpt", "ChatGPT"), ("chatgpt", "ChatGPT"), ("o1", "ChatGPT"), ("o3", "ChatGPT"), ("o4", "ChatGPT")]


def addressee_for(model: str) -> str | None:
    """Default greeting name for a model id (None if the provider is not recognised; pass --addressee then)."""
    m = model.lower().split("/")[-1]  # tolerate "models/gemini-..." and "openai/gpt-..." style ids
    for prefix, name in ADDRESSEE_BY_PREFIX:
        if m.startswith(prefix):
            return name
    return None


def readdress(prompt: str, name: str) -> str:
    """The same prompt greeting `name` ("Hello Claude, ..." -> "Hello Gemini, ...")."""
    if name == ADDRESSEE:
        return prompt
    head = f"Hello {ADDRESSEE},"
    assert prompt.startswith(head), prompt[:40]
    return f"Hello {name}," + prompt[len(head):]


def trailer(answer_space) -> str:
    return PROMPT["trailer"].format(phrase=answer_phrase(answer_space))


def render(rules: list[str], input_text: str, answer_space, numbered: bool = True) -> str:
    body = "\n".join(f"{i + 1}. {r}" if numbered else r for i, r in enumerate(rules))
    return f"{PROMPT['header']}\n\nRules:\n{body}\n\n{trailer(answer_space)}\n\nInput: {input_text}"


def item(*, id_, family, rules, input_text, answer, answer_space, difficulty, rationale,
         kind="letter", representation="none", tags=None, numbered=True):
    assert answer in answer_space, (answer, answer_space)
    return {
        "id": id_,
        "family": family,
        "source": "generated",
        "prompt": render(rules, input_text, answer_space, numbered),
        "rules": rules,
        "input": input_text,
        "answer": answer,
        "answer_space": list(answer_space),
        "difficulty": difficulty,
        "representation_dependence": representation,
        "rationale": rationale,
        "tags": tags or [],
    }


# --------------------------------------------------------------------------
# Family: chain_lookup  (compose k in-context tables)
# --------------------------------------------------------------------------
CHAIN_VOCAB = [
    ["dog", "cat", "cow", "pig", "hen", "fox", "owl", "bee", "ant", "eel"],
    ["red", "blue", "green", "black", "white", "pink", "gray", "brown", "olive", "cyan"],  # "gold" moved out: it is also a metal (layer 8)
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
]


def gen_chain_lookup(rng: random.Random, id_: str, table_size: int, hops: int, want_answer: str | None = None):
    assert 1 <= hops <= len(CHAIN_VOCAB) and table_size <= 10
    # Layers must be pairwise disjoint: a word in two layers appears twice as a key, and a reader resolving it by
    # textual position gets a wrong hop for reasons unrelated to depth ("gold" was a colour and a metal until 2026-09-06).
    assert len({w for v in CHAIN_VOCAB for w in v}) == sum(len(v) for v in CHAIN_VOCAB), "CHAIN_VOCAB layers overlap"
    layers = [rng.sample(v, table_size) for v in CHAIN_VOCAB[:hops]]
    digits = rng.sample("0123456789", table_size)
    layers.append(digits)
    # Wording signed off by the user on 2026-09-06; do not edit without approval.
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
    if want_answer is None:
        x = rng.choice(layers[0])
    else:  # every table is a bijection, so exactly one input word lands on the wanted digit: walk back through the tables
        x = want_answer
        for m in reversed(maps):
            x = {v: k for k, v in m.items()}[x]
    path = [x]
    for m in maps:
        x = m[x]
        path.append(x)
    return item(
        id_=id_, family="chain_lookup", rules=rules, input_text=path[0], answer=path[-1],
        answer_space=sorted(digits), kind="digit",
        difficulty={"hops": hops, "n_rules": hops * table_size, "world_knowledge": "none"},
        rationale=" -> ".join(path), tags=["multi_hop", "in_context_only"],
    )


# --------------------------------------------------------------------------
# Family: state_machine  (simulate a DFA over an input string)
# --------------------------------------------------------------------------
def gen_state_machine(rng: random.Random, id_: str, n_states: int, seq_len: int, alphabet: str = "ab",
                      want_answer: str | None = None, no_repeat: bool = True):
    """Simulate a permutation automaton. The INPUT is the start state; the symbol string lives in the rules, so the
    end state is a bijection of the start state and can be stratified over the states (want_answer).

    2026-09-06 audit: symbols were fixed-point-free permutations, which made every 1-step answer impossible (partial
    simulation scored BELOW chance and "exclude the one-step states" above it); they are now uniform permutations.
    Rejected: a symbol that is the identity, two symbols with the same permutation, a composite that is the identity
    (answer == input for every start), and single-symbol strings when the string is short.

    Repeated symbols (2026-09-06, evening): with three symbols the string often ends in "a a", and a permutation returns
    a state after two identical steps with probability 2/5, so "the state two symbols before the end" equalled the gold
    25-27% of the time at every length (chance 20%): a partial simulator that stops early got a structural bonus.
    With six symbols and no immediate repeats (no_repeat) every intermediate state coincides with the gold within
    0.01 of chance in simulation; the ladder uses that setting."""
    states = list(range(n_states))
    for _ in range(1000):
        trans, perms = {}, []
        for ch in alphabet:
            perm = states[:]
            rng.shuffle(perm)
            perms.append(tuple(perm))
            for s in states:
                trans[(s, ch)] = perm[s]
        if len(set(perms)) < len(alphabet) or tuple(states) in perms:
            continue
        seq: list[str] = []
        for _ in range(seq_len):
            ch = rng.choice(alphabet)
            while no_repeat and seq and ch == seq[-1]:
                ch = rng.choice(alphabet)
            seq.append(ch)
        if 1 < seq_len <= 6 and len(set(seq)) < 2:
            continue
        g = list(states)  # g[s] = state reached from s after the whole string
        for ch in seq:
            g = [trans[(s, ch)] for s in g]
        if g == states:
            continue
        break
    target = int(want_answer) if want_answer is not None else rng.randrange(n_states)
    s0 = g.index(target)
    # Wording signed off by the user on 2026-09-06; do not edit without approval. The word is written in composition
    # order (rightmost permutation applied first), so it is the reverse of the application sequence `seq`.
    number = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}[len(alphabet)]
    names = ", ".join(alphabet[:-1]) + f" and {alphabet[-1]}"
    rules = [f"This question concerns a set of {n_states} states, numbered 0-{n_states - 1}, and {number} permutations of "
             f"those states, named {names}."]
    for ch in alphabet:
        rules.append(f"{ch} is given by: " + ", ".join(f"{ch}({s}) = {trans[(s, ch)]}" for s in states) + ".")
    rules.append("When I write a b c (x), I mean a(b(c(x))): the rightmost permutation is applied first.")
    rules.append("Let x_0 be the input state. Answer with " + " ".join(reversed(seq)) + " (x_0).")
    s, path = s0, [s0]
    for ch in seq:
        s = trans[(s, ch)]
        path.append(s)
    assert s == target
    return item(
        id_=id_, family="state_machine", rules=rules, input_text=str(s0), answer=str(s),
        answer_space=[str(t) for t in states], kind="digit",
        difficulty={"steps": seq_len, "n_rules": n_states * len(alphabet), "world_knowledge": "none"},
        rationale="states visited: " + " ".join(map(str, path)), tags=["simulation", "sequential"],
    )


# --------------------------------------------------------------------------
# Family: sequence_mod  (two-term recurrence through an in-context table, k terms on)
# --------------------------------------------------------------------------
# History. v1 iterated x <- (m*x + c) mod 10: f^4 = identity for every m coprime to 10. v2/v3 used the Fibonacci
# recurrence t_n = t_{n-1} + t_{n-2} mod 10: it is LINEAR, so t_n = F(n-2)*t1 + F(n-1)*t2 with coefficients every
# model knows, and at 20 steps the answer was literally (input + 5) mod 10 for every item (2026-09-06 audit).
# Every polynomial recurrence mod 10 splits by CRT into a tiny mod-2 automaton and degenerates the same way.
# The recurrence now goes through a random digit table T stated in the rules: t_n = (T[t_{n-1}] + t_{n-2}) mod 10.
# The pair map (a, b) -> (b, T[b] + a) is a bijection, so orbits never collapse; items whose orbit returns to its
# starting pair before the answer term are rejected.
def gen_sequence_mod(rng: random.Random, id_: str, steps: int, want_answer: str | None = None):
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
            if ok and (want_answer is None or str(path[-1]) == want_answer):
                cands.append((x0, path))
        if cands:
            x0, path = rng.choice(cands)
            break
    else:
        raise RuntimeError(id_)
    # Wording signed off by the user on 2026-09-06; do not edit without approval.
    rules = ['This question concerns a sequence whose elements are digits from 0-9. We write "term 1" for the first '
             'element, "term 2" for the second, and so on.',
             f"Term 1 is {a0}. Term 2 is the input digit.",
             "Let f be the following permutation of the digits 0-9: " + ", ".join(f"f({d}) = {T[d]}" for d in range(10)) + ".",
             "For every n >= 3, term n = (f(term (n-1)) + term (n-2)) mod 10.",
             f"Answer with term {steps + 2}."]
    return item(
        id_=id_, family="sequence_mod", rules=rules, input_text=str(x0), answer=str(path[-1]),
        answer_space=[str(i) for i in range(10)], kind="digit",
        difficulty={"steps": steps, "n_rules": 4, "operand_magnitude": 10, "world_knowledge": "none"},
        rationale=f"terms 1..{steps + 2}: {a0} " + " ".join(map(str, path)), tags=["iteration", "in_context_only", "two_state"],
    )


# --------------------------------------------------------------------------
# Family: iterate_map  (one digit->digit table applied k times)
# --------------------------------------------------------------------------
def gen_iterate_map(rng: random.Random, id_: str, steps: int, want_answer: str | None = None):
    """Function iteration with a single in-context table. The table is ONE cycle over the 26 capital letters, so no
    orbit revisits a letter within 25 steps (a random permutation has short cycles on which f^k collapses to the
    identity, the same trap as the affine map in the first sequence_mod).

    Cycle length (2026-09-06 audit): on a 10-cycle f^9 = f^-1, so "9 steps" was ONE backward table lookup and
    7 steps were three; the effective depth of a k-step item on an n-cycle is min(k, n-k). A single cycle of
    length >= 2*max_steps is the only fix (two cycles of 13 reintroduce it at k=12), hence 26 letters and k <= 13.
    Tables with 3+ alphabetical-neighbour entries (f(L) = L+1 or L-1) are redrawn so a Caesar-shift prior has
    nothing to work with (a random 26-cycle has about one such entry)."""
    assert 1 <= steps <= 13
    n = len(LETTERS)
    for _ in range(200):
        cyc = list(LETTERS)
        rng.shuffle(cyc)
        f = {cyc[i]: cyc[(i + 1) % n] for i in range(n)}
        if sum(abs(LETTERS.index(f[c]) - LETTERS.index(c)) == 1 for c in LETTERS) < 3:
            break
    if want_answer is None:
        x0 = rng.choice(LETTERS)
    else:  # the table is a bijection: walk back k steps from the wanted letter
        inv = {v: k for k, v in f.items()}
        x0 = want_answer
        for _ in range(steps):
            x0 = inv[x0]
    x, path = x0, [x0]
    for _ in range(steps):
        x = f[x]
        path.append(x)
    # Wording signed off by the user on 2026-09-06; do not edit without approval.
    rules = ["This question concerns a permutation g of the capital letters A-Z, given by: "
             + ", ".join(f"g({c}) = {f[c]}" for c in LETTERS) + ".",
             f"Apply g to the input letter exactly {steps} time{'s' if steps > 1 else ''} in succession.",
             "Answer with the letter you end on."]
    return item(
        id_=id_, family="iterate_map", rules=rules, input_text=x0, answer=x,
        answer_space=list(LETTERS), kind="letter",
        difficulty={"steps": steps, "n_rules": 3, "world_knowledge": "none"},
        rationale="orbit: " + " -> ".join(path), tags=["iteration", "in_context_only"],
    )


# --------------------------------------------------------------------------
# Difficulty ladders
# --------------------------------------------------------------------------
# Ladder v2 (2026-09-05 evening). ONE knob per family, monotone in level; everything else is fixed
# and stated in the comment. Levels 1-5 are the measurement ladder (20 items each by default);
# levels 6-7 are "hard" rungs (60 items each) chosen so Opus 5 with 100 filler tokens is at chance,
# or as far as the family can be pushed if it never gets there. KNOB names the knob for each family;
# generate_all copies it into difficulty.knob / difficulty.knob_value so labels cannot be ambiguous.
# The v1 ladders (mixed knobs and variants, 3 items per level) are archived in results/data_v1/.
HARD_FROM_LEVEL = 6


KNOB = {
    "sequence_mod": "steps",             # two-term recurrence through a stated digit table, explicit term numbers
    "iterate_map": "steps",              # one letter->letter table (a 26-cycle) applied k times
    "state_machine": "steps",            # 5 states, 6 symbols (no immediate repeats), uniform permutations; input = start state
    "chain_lookup": "hops",              # tables of 10 entries, input = one word
}


# Aligned levels (2026-09-06 night): a level is a DEPTH on a grid shared by every family, so level k means the same
# number of compositions everywhere; a family simply lacks the levels whose depth it cannot construct.
DEPTHS = [1, 2, 3, 4, 6, 8, 12, 16, 20]


LEVEL_OF = {d: i + 1 for i, d in enumerate(DEPTHS)}


def ladder(make_kw, depths):
    return [(LEVEL_OF[d], make_kw(d)) for d in depths]


LADDERS: dict[str, list[tuple[int, dict]]] = {
    "sequence_mod": ladder(lambda d: dict(steps=d), [1, 2, 3, 4, 6, 8, 12, 16, 20]),
    "iterate_map": ladder(lambda d: dict(steps=d), [1, 2, 3, 4, 6, 8, 12]),  # 26-cycle: depth is honest only for k <= 13
    "state_machine": ladder(lambda d: dict(n_states=5, seq_len=d, alphabet="abcdef"), [1, 2, 3, 4, 6, 8, 12, 16, 20]),
    "chain_lookup": ladder(lambda d: dict(table_size=10, hops=d), [1, 2, 3, 4, 6, 8, 12, 16]),  # 16 vocabulary layers
}


# Fixed answer spaces, for the per-level stratification of the gold answer in generate_all.
ANSWER_SPACE = {"sequence_mod": [str(i) for i in range(10)], "iterate_map": list(LETTERS),
                "state_machine": [str(i) for i in range(5)], "chain_lookup": [str(i) for i in range(10)]}


GENERATORS = {
    "chain_lookup": gen_chain_lookup, "state_machine": gen_state_machine,
    "sequence_mod": gen_sequence_mod, "iterate_map": gen_iterate_map,
}


def stratified_targets(rng: random.Random, space: list[str], n: int) -> list[str]:
    """Each answer as equally often as n allows (n = 20 over 26 letters -> 20 distinct letters), in random order."""
    pool = list(space) * (-(-n // len(space)))
    rng.shuffle(pool)
    return pool[:n]


def generate_all(seed: int, per_level: int, hard_per_level: int | None = None) -> list[dict]:
    out = []
    for fam, ladder in LADDERS.items():
        for level, kw in ladder:
            n = hard_per_level if (hard_per_level is not None and level >= HARD_FROM_LEVEL) else per_level
            targets = stratified_targets(random.Random(f"{seed}-{fam}-{level}-targets"), ANSWER_SPACE[fam], n)
            for i in range(n):
                rng = random.Random(f"{seed}-{fam}-{level}-{i}")
                kw2 = dict(kw, want_answer=targets[i])  # gold is uniform per level by construction
                it = GENERATORS[fam](rng, f"{fam}-L{level}-s{seed}-{i}", **kw2)
                it["difficulty"]["level"] = level
                it["difficulty"]["knob"] = KNOB[fam]
                it["difficulty"]["knob_value"] = it["difficulty"][KNOB[fam]]
                out.append(it)
    return out


def validate(items: list[dict], schema_path: str = "schema.json") -> None:
    schema = json.load(open(schema_path, encoding="utf-8"))
    req = set(schema["required"])
    fams = set(schema["properties"]["family"]["enum"])
    pat = re.compile(schema["properties"]["answer"]["pattern"])
    ids = set()
    for it in items:
        missing = req - set(it)
        assert not missing, (it["id"], missing)
        assert it["family"] in fams, it["family"]
        assert pat.match(it["answer"]), it["answer"]
        assert it["answer"] in it["answer_space"], it["id"]
        assert it["id"] not in ids, it["id"]
        ids.add(it["id"])
        assert it["prompt"].rstrip().endswith(it["input"]), it["id"]


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
    from collections import Counter
    print(f"wrote {len(items)} items to {a.out}")
    for fam, n in Counter(it["family"] for it in items).items():
        print(f"  {fam}: {n}")
