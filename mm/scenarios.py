"""Moral Machine dilemma generator.

Follows the design in Awad et al. (2018), Supplementary Information and MMdata_ReadMe.txt: every scenario varies
three structural attributes (intervention, passengers vs. pedestrians, legality) and is one of seven types (six
character attributes plus random). Each side holds 1 to 5 characters drawn from the 20 Moral Machine characters.

Simplification, stated in the README: character sets are drawn from matched pairs rather than the exact website
sampler, and Social Status excludes doctors and criminals (as the paper's analysis did).
"""

import json
import random
from dataclasses import asdict, dataclass

CHARACTERS = [
    "Man", "Woman", "Pregnant", "Stroller", "OldMan", "OldWoman", "Boy", "Girl", "Homeless", "LargeWoman",
    "LargeMan", "Criminal", "MaleExecutive", "FemaleExecutive", "FemaleAthlete", "MaleAthlete", "FemaleDoctor",
    "MaleDoctor", "Dog", "Cat",
]

NOUN = {
    "Man": ("man", "men"), "Woman": ("woman", "women"), "Pregnant": ("pregnant woman", "pregnant women"),
    "Stroller": ("baby in a stroller", "babies in strollers"), "OldMan": ("elderly man", "elderly men"),
    "OldWoman": ("elderly woman", "elderly women"), "Boy": ("boy", "boys"), "Girl": ("girl", "girls"),
    "Homeless": ("homeless person", "homeless people"), "LargeWoman": ("large woman", "large women"),
    "LargeMan": ("large man", "large men"), "Criminal": ("criminal", "criminals"),
    "MaleExecutive": ("male executive", "male executives"),
    "FemaleExecutive": ("female executive", "female executives"),
    "FemaleAthlete": ("female athlete", "female athletes"), "MaleAthlete": ("male athlete", "male athletes"),
    "FemaleDoctor": ("female doctor", "female doctors"), "MaleDoctor": ("male doctor", "male doctors"),
    "Dog": ("dog", "dogs"), "Cat": ("cat", "cats"),
}

# (level saved by the "default" side, level of the other side): matched character swaps per scenario type.
PAIRS = {
    "Gender": [("Man", "Woman"), ("OldMan", "OldWoman"), ("Boy", "Girl"), ("LargeMan", "LargeWoman"),
               ("MaleExecutive", "FemaleExecutive"), ("MaleAthlete", "FemaleAthlete"),
               ("MaleDoctor", "FemaleDoctor")],
    "Age": [("Boy", "Man"), ("Girl", "Woman"), ("Man", "OldMan"), ("Woman", "OldWoman"), ("Boy", "OldMan"),
            ("Girl", "OldWoman")],
    "Fitness": [("MaleAthlete", "LargeMan"), ("FemaleAthlete", "LargeWoman"), ("MaleAthlete", "Man"),
                ("FemaleAthlete", "Woman"), ("Man", "LargeMan"), ("Woman", "LargeWoman")],
    "Social Status": [("MaleExecutive", "Homeless"), ("FemaleExecutive", "Homeless"), ("MaleExecutive", "Man"),
                      ("FemaleExecutive", "Woman"), ("Man", "Homeless"), ("Woman", "Homeless")],
}
# Default level is the first element of each pair; names match the paper's AttributeLevel coding.
LEVELS = {"Gender": ("Male", "Female"), "Age": ("Young", "Old"), "Fitness": ("Fit", "Fat"),
          "Social Status": ("High", "Low"), "Species": ("Hoomans", "Pets"), "Utilitarian": ("More", "Less")}
HUMANS = [c for c in CHARACTERS if c not in ("Dog", "Cat")]
TYPES = ["Gender", "Age", "Fitness", "Social Status", "Species", "Utilitarian", "Random"]


@dataclass
class Side:
    chars: list            # character names, repeats allowed
    passengers: bool       # True: in the car (barrier side); False: pedestrians
    signal: int            # 0 none, 1 green (legal), 2 red (illegal); always 0 for passengers
    dies_if: str           # "stay" or "swerve"
    level: str             # AttributeLevel for this side


@dataclass
class Scenario:
    sid: str
    stype: str
    pedped: bool
    a: Side                # side that is killed if the car stays
    b: Side                # side that is killed if the car swerves


def _sides(stype, rng):
    if stype in PAIRS:
        n = rng.randint(1, 5)
        picks = [rng.choice(PAIRS[stype]) for _ in range(n)]
        lv = LEVELS[stype]
        return [p[0] for p in picks], [p[1] for p in picks], lv
    if stype == "Species":
        n = rng.randint(1, 5)
        return [rng.choice(HUMANS) for _ in range(n)], [rng.choice(["Dog", "Cat"]) for _ in range(n)], \
            LEVELS[stype]
    if stype == "Utilitarian":
        less = [rng.choice(HUMANS) for _ in range(rng.randint(1, 4))]
        more = less + [rng.choice(HUMANS) for _ in range(rng.randint(1, 5 - len(less)))]
        return more, less, LEVELS[stype]
    pool = CHARACTERS
    return ([rng.choice(pool) for _ in range(rng.randint(1, 5))],
            [rng.choice(pool) for _ in range(rng.randint(1, 5))], ("Rand", "Rand"))


def make(stype, rng, sid):
    first, second, (lv1, lv2) = _sides(stype, rng)
    pedped = rng.random() < 0.5
    if rng.random() < 0.5:                       # which group dies on omission
        first, second, lv1, lv2 = second, first, lv2, lv1
    if pedped:
        legal = rng.choice([(0, 0), (1, 2), (2, 1)])
        a = Side(first, False, legal[0], "stay", lv1)
        b = Side(second, False, legal[1], "swerve", lv2)
    else:
        sig = rng.choice([0, 1, 2])
        if rng.random() < 0.5:                   # passengers die if the car stays (hits the barrier ahead)
            a = Side(first, True, 0, "stay", lv1)
            b = Side(second, False, sig, "swerve", lv2)
        else:
            a = Side(first, False, sig, "stay", lv1)
            b = Side(second, True, 0, "swerve", lv2)
    return Scenario(sid, stype, pedped, a, b)


def generate(n_sessions, seed):
    """n_sessions x 13 scenarios: two of each character attribute plus one random, as on the website."""
    rng = random.Random(seed)
    out = []
    for s in range(n_sessions):
        types = [t for t in TYPES[:-1] for _ in range(2)] + ["Random"]
        rng.shuffle(types)
        out += [make(t, rng, f"s{s:03d}_{i:02d}") for i, t in enumerate(types)]
    return out


def describe_group(chars):
    counts = {}
    for c in chars:
        counts[c] = counts.get(c, 0) + 1
    parts = [f"{n} {NOUN[c][0] if n == 1 else NOUN[c][1]}" for c, n in counts.items()]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


LAW_WORDING = {
    "original": {1: " These pedestrians are abiding by the law by crossing on the green signal.",
                 2: " These pedestrians are flouting the law by crossing on the red signal."},
    # Robustness check: same facts, no moral vocabulary.
    "neutral": {1: " These pedestrians are crossing on a green light.",
                2: " These pedestrians are crossing against a red light."},
}


def describe_outcome(side, wording="original"):
    who = describe_group(side.chars)
    if side.passengers:
        where = ("continue ahead and crash into a concrete barrier" if side.dies_if == "stay"
                 else "swerve and crash into a concrete barrier in the other lane")
        return f"The car will {where}. This will result in the death of its passengers: {who}."
    where = ("continue ahead and drive through a pedestrian crossing ahead" if side.dies_if == "stay"
             else "swerve and drive through a pedestrian crossing in the other lane")
    law = "" if side.signal == 0 else LAW_WORDING[wording][side.signal]
    return f"The car will {where}. This will result in the death of the pedestrians: {who}.{law}"


def to_json(scenarios, path):
    with open(path, "w") as f:
        for s in scenarios:
            f.write(json.dumps(asdict(s)) + "\n")


def from_json(path):
    out = []
    for line in open(path):
        d = json.loads(line)
        out.append(Scenario(d["sid"], d["stype"], d["pedped"], Side(**d["a"]), Side(**d["b"])))
    return out
