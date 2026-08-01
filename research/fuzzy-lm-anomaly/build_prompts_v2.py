"""Stage 1b -- template-matched probe set (supersedes build_prompts.py).

Why this exists. The v1 false-premise probes were templated while the truthful
comparison set (TriviaQA) was not, so a detector could separate them on prompt
style rather than on fabrication. Length was controlled in §8; template novelty
was not, and it is the last confound standing between the §9 result and a claim
about hallucination.

Fix: every fabricated question has a **real-entity twin in the identical surface
form**, drawn from a curated table of facts with programmatically checkable
answers.

    real : "What is the capital city of Portugal?"      -> gradeable, truthful
    fake : "What is the capital city of Brazendia?"     -> necessarily fabricated

Five template families, each with ~30 curated real instances and invented twins.
Because the two sides share the template exactly, the evaluation can match on
(template, n_tokens) and any surviving separation is neither length nor style.

Emitted families:
  triviaqa       closed-book QA (unchanged; the in-distribution control)
  template_real  real entity, gradeable  -> supplies known-good behaviour
  template_fake  invented entity         -> necessarily fabricated
  falsepremise   the original v1 probes, kept for continuity with §7-§9

Every row carries a `template` field so downstream matching can use it.
"""

import argparse
import itertools
import json
import random
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).parent / "data"

# --------------------------------------------------------------------------
# Curated real facts. Kept to well-established, unambiguous items so that
# alias matching is reliable without a knowledge base.
# --------------------------------------------------------------------------
def _parse(spec):
    """'Key: a | b ; Key2: c' -> {'Key': ['a','b'], 'Key2': ['c']}.

    Compact literal form; keeps ~200 curated facts readable.
    """
    out = {}
    for item in spec.strip().split(";"):
        if not item.strip():
            continue
        k, v = item.split(":", 1)
        out[k.strip()] = [a.strip() for a in v.split("|") if a.strip()]
    return out


# Contested or dual capitals are omitted rather than guessed at.
CAPITALS = _parse("""
France: Paris; Japan: Tokyo; Canada: Ottawa; Australia: Canberra;
Brazil: Brasilia | Brasília; Egypt: Cairo; Kenya: Nairobi; Norway: Oslo;
Portugal: Lisbon | Lisboa; Thailand: Bangkok; Peru: Lima;
Poland: Warsaw | Warszawa; Greece: Athens; Vietnam: Hanoi; Morocco: Rabat;
Chile: Santiago; Sweden: Stockholm; Hungary: Budapest; Ireland: Dublin;
Nepal: Kathmandu; Cuba: Havana; Iceland: Reykjavik | Reykjavík;
Finland: Helsinki; Austria: Vienna | Wien; Turkey: Ankara;
Argentina: Buenos Aires; Mexico: Mexico City; India: New Delhi | Delhi;
Indonesia: Jakarta; Ethiopia: Addis Ababa; Germany: Berlin; Italy: Rome | Roma;
Spain: Madrid; Netherlands: Amsterdam; Belgium: Brussels; Switzerland: Bern;
Denmark: Copenhagen; Russia: Moscow; Ukraine: Kyiv | Kiev; Romania: Bucharest;
Bulgaria: Sofia; Serbia: Belgrade; Croatia: Zagreb; Slovakia: Bratislava;
Slovenia: Ljubljana; Czechia: Prague; Estonia: Tallinn; Latvia: Riga;
Lithuania: Vilnius; Belarus: Minsk; China: Beijing; South Korea: Seoul;
North Korea: Pyongyang; Philippines: Manila; Malaysia: Kuala Lumpur;
Myanmar: Naypyidaw; Cambodia: Phnom Penh; Laos: Vientiane; Bangladesh: Dhaka;
Pakistan: Islamabad; Afghanistan: Kabul; Iran: Tehran; Iraq: Baghdad;
Syria: Damascus; Lebanon: Beirut; Jordan: Amman; Saudi Arabia: Riyadh;
Yemen: Sanaa; Oman: Muscat; Qatar: Doha; Kuwait: Kuwait City;
United Arab Emirates: Abu Dhabi; Georgia: Tbilisi; Armenia: Yerevan;
Azerbaijan: Baku; Kazakhstan: Astana; Uzbekistan: Tashkent;
Mongolia: Ulaanbaatar; Nigeria: Abuja; Ghana: Accra; Senegal: Dakar;
Mali: Bamako; Sudan: Khartoum; Tanzania: Dodoma; Uganda: Kampala;
Zambia: Lusaka; Zimbabwe: Harare; Botswana: Gaborone; Namibia: Windhoek;
Angola: Luanda; Mozambique: Maputo; Madagascar: Antananarivo; Tunisia: Tunis;
Algeria: Algiers; Libya: Tripoli; Colombia: Bogota | Bogotá;
Venezuela: Caracas; Ecuador: Quito; Paraguay: Asuncion | Asunción;
Uruguay: Montevideo; Panama: Panama City; Costa Rica: San Jose | San José;
Guatemala: Guatemala City; Honduras: Tegucigalpa; Nicaragua: Managua;
Jamaica: Kingston; Dominican Republic: Santo Domingo;
Haiti: Port-au-Prince; New Zealand: Wellington; Fiji: Suva;
Papua New Guinea: Port Moresby
""")

ELEMENTS = _parse("""
Hydrogen: H; Helium: He; Lithium: Li; Beryllium: Be; Boron: B; Carbon: C;
Nitrogen: N; Oxygen: O; Fluorine: F; Neon: Ne; Sodium: Na; Magnesium: Mg;
Aluminium: Al; Silicon: Si; Phosphorus: P; Sulfur: S; Chlorine: Cl; Argon: Ar;
Potassium: K; Calcium: Ca; Scandium: Sc; Titanium: Ti; Vanadium: V;
Chromium: Cr; Manganese: Mn; Iron: Fe; Cobalt: Co; Nickel: Ni; Copper: Cu;
Zinc: Zn; Gallium: Ga; Germanium: Ge; Arsenic: As; Selenium: Se; Bromine: Br;
Krypton: Kr; Rubidium: Rb; Strontium: Sr; Yttrium: Y; Zirconium: Zr;
Niobium: Nb; Molybdenum: Mo; Technetium: Tc; Ruthenium: Ru; Rhodium: Rh;
Palladium: Pd; Silver: Ag; Cadmium: Cd; Indium: In; Tin: Sn; Antimony: Sb;
Tellurium: Te; Iodine: I; Xenon: Xe; Caesium: Cs; Barium: Ba; Lanthanum: La;
Cerium: Ce; Neodymium: Nd; Samarium: Sm; Europium: Eu; Gadolinium: Gd;
Terbium: Tb; Hafnium: Hf; Tantalum: Ta; Tungsten: W; Rhenium: Re; Osmium: Os;
Iridium: Ir; Platinum: Pt; Gold: Au; Mercury: Hg; Thallium: Tl; Lead: Pb;
Bismuth: Bi; Polonium: Po; Radon: Rn; Radium: Ra; Thorium: Th; Uranium: U;
Plutonium: Pu
""")
NOVELS = {
    "Pride and Prejudice": ["Jane Austen", "Austen"],
    "Moby-Dick": ["Herman Melville", "Melville"],
    "Great Expectations": ["Charles Dickens", "Dickens"],
    "Jane Eyre": ["Charlotte Bronte", "Charlotte Brontë", "Bronte"],
    "Wuthering Heights": ["Emily Bronte", "Emily Brontë", "Bronte"],
    "The Great Gatsby": ["F. Scott Fitzgerald", "Fitzgerald"],
    "Brave New World": ["Aldous Huxley", "Huxley"],
    "Nineteen Eighty-Four": ["George Orwell", "Orwell"],
    "Animal Farm": ["George Orwell", "Orwell"],
    "To Kill a Mockingbird": ["Harper Lee"],
    "The Old Man and the Sea": ["Ernest Hemingway", "Hemingway"],
    "Crime and Punishment": ["Fyodor Dostoevsky", "Dostoevsky", "Dostoyevsky"],
    "War and Peace": ["Leo Tolstoy", "Tolstoy"],
    "Anna Karenina": ["Leo Tolstoy", "Tolstoy"],
    "Madame Bovary": ["Gustave Flaubert", "Flaubert"],
    "Les Miserables": ["Victor Hugo", "Hugo"],
    "Don Quixote": ["Miguel de Cervantes", "Cervantes"],
    "The Trial": ["Franz Kafka", "Kafka"],
    "Ulysses": ["James Joyce", "Joyce"],
    "Mrs Dalloway": ["Virginia Woolf", "Woolf"],
    "Frankenstein": ["Mary Shelley", "Shelley"],
    "Dracula": ["Bram Stoker", "Stoker"],
    "The Hobbit": ["J. R. R. Tolkien", "Tolkien"],
    "Lord of the Flies": ["William Golding", "Golding"],
    "Catch-22": ["Joseph Heller", "Heller"],
    "Slaughterhouse-Five": ["Kurt Vonnegut", "Vonnegut"],
    "Beloved": ["Toni Morrison", "Morrison"],
    "One Hundred Years of Solitude": ["Gabriel Garcia Marquez",
                                      "Gabriel García Márquez", "Marquez"],
    "Things Fall Apart": ["Chinua Achebe", "Achebe"],
    "The Catcher in the Rye": ["J. D. Salinger", "Salinger"],
}
CURRENCIES = {
    "Japan": ["yen"], "India": ["rupee"], "Russia": ["ruble", "rouble"],
    "Poland": ["zloty", "złoty"], "Sweden": ["krona"], "Denmark": ["krone"],
    "Switzerland": ["franc"], "China": ["yuan", "renminbi"],
    "South Korea": ["won"], "Vietnam": ["dong", "đồng"], "Thailand": ["baht"],
    "Turkey": ["lira"], "Israel": ["shekel"], "Mexico": ["peso"],
    "Brazil": ["real"], "Peru": ["sol"], "South Africa": ["rand"],
    "Nigeria": ["naira"], "Kenya": ["shilling"], "Morocco": ["dirham"],
    "Hungary": ["forint"], "Iceland": ["krona", "króna"],
    "Malaysia": ["ringgit"], "Indonesia": ["rupiah"], "Bangladesh": ["taka"],
    "Saudi Arabia": ["riyal"], "Norway": ["krone"], "Czechia": ["koruna"],
    "Egypt": ["pound"], "Philippines": ["peso"],
}
FILMS = {
    "Jaws": ["Steven Spielberg", "Spielberg"],
    "Psycho": ["Alfred Hitchcock", "Hitchcock"],
    "Vertigo": ["Alfred Hitchcock", "Hitchcock"],
    "Rear Window": ["Alfred Hitchcock", "Hitchcock"],
    "Citizen Kane": ["Orson Welles", "Welles"],
    "Pulp Fiction": ["Quentin Tarantino", "Tarantino"],
    "Goodfellas": ["Martin Scorsese", "Scorsese"],
    "Taxi Driver": ["Martin Scorsese", "Scorsese"],
    "The Godfather": ["Francis Ford Coppola", "Coppola"],
    "Apocalypse Now": ["Francis Ford Coppola", "Coppola"],
    "Alien": ["Ridley Scott", "Scott"],
    "Blade Runner": ["Ridley Scott", "Scott"],
    "Titanic": ["James Cameron", "Cameron"],
    "The Terminator": ["James Cameron", "Cameron"],
    "Inception": ["Christopher Nolan", "Nolan"],
    "Memento": ["Christopher Nolan", "Nolan"],
    "The Shining": ["Stanley Kubrick", "Kubrick"],
    "2001: A Space Odyssey": ["Stanley Kubrick", "Kubrick"],
    "Metropolis": ["Fritz Lang", "Lang"],
    "Seven Samurai": ["Akira Kurosawa", "Kurosawa"],
    "Rashomon": ["Akira Kurosawa", "Kurosawa"],
    "Amarcord": ["Federico Fellini", "Fellini"],
    "Persona": ["Ingmar Bergman", "Bergman"],
    "Do the Right Thing": ["Spike Lee"],
    "Parasite": ["Bong Joon-ho", "Bong"],
    "Spirited Away": ["Hayao Miyazaki", "Miyazaki"],
    "Get Out": ["Jordan Peele", "Peele"],
    "Mad Max: Fury Road": ["George Miller", "Miller"],
    "Fargo": ["Joel Coen", "Coen"],
    "Jurassic Park": ["Steven Spielberg", "Spielberg"],
}

# --------------------------------------------------------------------------
# Invented entities, generated combinatorially and addressed by index so that
# uniqueness is guaranteed by construction. Rejection sampling from a small
# hand-written pool silently under-delivers (it capped at 411 of 5,000), which
# would leave the fake side of each template too thin to match against.
# --------------------------------------------------------------------------
_P_A = ["Braz", "Kelm", "Ondas", "Vuren", "Tashm", "Grell", "Ospr", "Mordaq",
        "Zelbr", "Havr", "Pusc", "Wexl", "Ordr", "Salv", "Tyren", "Quiv",
        "Nerb", "Cadr", "Merlu", "Threq"]
# Distinctive multi-character infixes and suffixes: short ones (e.g. "Braz"+"or"
# +"ia") collide with real places like Brazoria County, which would break the
# "necessarily fabricated" label.
_P_B = ["end", "ond", "arv", "esk", "ulth", "ovr", "imn", "aqu"]
_P_C = ["ia", "ania", "esia", "landia", "morra", "stania", "voria", "dora"]

_E_A = ["Vandal", "Threqu", "Oscad", "Merluv", "Kaspher", "Ytterb", "Drennic",
        "Palvor", "Zirbal", "Norvex", "Cadril", "Wexlar", "Ombrath", "Quenth",
        "Selvan", "Turmac", "Behrol", "Ganvis", "Praxil", "Emberd", "Corvan",
        "Nystral", "Deltrav", "Halcyr"]
_E_B = ["a", "o", "e", "i", "u", "y"]
_E_C = ["ium", "ine", "on", "ide", "ate", "yl", "ane"]

_T_PROPER = ["Vastrimer", "Ombrathe", "Thrennick", "Belhoun", "Karsivett",
             "Dunmorrow", "Ashkelvane", "Prendaux", "Wexlingham", "Silvenwood",
             "Corrathine", "Oquendel", "Marchevant", "Follenby", "Hastrimond",
             "Ravenshold", "Quillemont", "Tarrowgate", "Ellisbrand", "Nordhaven"]
_T_NOUN = ["Ledger", "Cartograph", "Reliquary", "Aviary", "Threnody", "Almanac",
           "Cipher", "Orrery", "Lantern", "Bequest", "Verdict", "Sonata",
           "Escarpment", "Interregnum", "Codex", "Meridian", "Testament",
           "Inheritance", "Conjecture", "Palisade"]
_T_PAT = ["The {} {}", "{}'s {}", "A {} {}"]


def _combo(i, a, b, c):
    """The i-th element of a x b x c, joined -- deterministic and collision-free."""
    return (a[i % len(a)] + b[(i // len(a)) % len(b)]
            + c[(i // (len(a) * len(b))) % len(c)])


def _fake_place(i):        # 20 x 8 x 8 = 1,280 distinct
    return _combo(i, _P_A, _P_B, _P_C)


def _fake_elem(i):         # 24 x 6 x 7 = 1,008 distinct
    return _combo(i, _E_A, _E_B, _E_C)


def _fake_title(i):        # 20 x 20 x 3 = 1,200 distinct
    p = _T_PROPER[i % len(_T_PROPER)]
    n = _T_NOUN[(i // len(_T_PROPER)) % len(_T_NOUN)]
    return _T_PAT[(i // (len(_T_PROPER) * len(_T_NOUN))) % len(_T_PAT)].format(p, n)


# Three surface forms per family, applied IDENTICALLY to the real and fake side,
# so the phrasing distribution is balanced by construction and matching on
# `template` alone is sufficient.
TEMPLATES = {
    "capital": (["What is the capital city of {}?",
                 "Name the capital of {}.",
                 "Which city serves as the capital of {}?"],
                CAPITALS, _fake_place),
    "symbol": (["What is the chemical symbol for the element {}?",
                "Give the chemical symbol of {}.",
                "Which symbol denotes the element {}?"],
               ELEMENTS, _fake_elem),
    "novel": (["Who wrote the novel {}?",
               "Name the author of the novel {}.",
               "Which novelist wrote {}?"],
              NOVELS, _fake_title),
    "currency": (["What is the currency of {}?",
                  "Name the official currency used in {}.",
                  "Which currency is used in {}?"],
                 CURRENCIES, _fake_place),
    "film": (["Who directed the film {}?",
              "Name the director of the film {}.",
              "Which director made the film {}?"],
             FILMS, _fake_title),
}


def build_template_pairs(n_real, n_fake, seed=0):
    """Real and invented instances of the *same* templates.

    Real instances cycle the curated table (each fact repeats n_real/|table|
    times, which is fine: greedy decoding makes repeats identical, so they are
    deduplicated to unique facts downstream). Fake instances are index-addressed
    and therefore unique.
    """
    rows = []
    per_fake = max(1, n_fake // len(TEMPLATES))

    for tname, (patterns, table, fake_fn) in TEMPLATES.items():
        # Real side: every curated fact x every phrasing. No duplicates, so no
        # wasted generations (greedy decoding makes a repeated prompt identical).
        for i, (k, pat) in enumerate((k, p) for k in table for p in patterns):
            rows.append({"id": f"tr-{tname}-{i:05d}", "family": "template_real",
                         "subtype": tname, "template": tname,
                         "phrasing": patterns.index(pat),
                         "question": pat.format(k), "answers": table[k]})
        # Fake side: the same phrasings, cycled over index-addressed entities.
        seen = set()
        for i in range(per_fake * 3):
            if len(seen) >= per_fake:
                break
            q = patterns[i % len(patterns)].format(fake_fn(i // len(patterns)))
            if q in seen:
                continue
            seen.add(q)
            rows.append({"id": f"tf-{tname}-{i:05d}", "family": "template_fake",
                         "subtype": tname, "template": tname,
                         "phrasing": i % len(patterns),
                         "question": q, "answers": []})
    return rows


def build_triviaqa(n, seed=0):
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext",
                      split="validation", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=20_000)
    rows = []
    for ex in ds:
        if len(rows) >= n:
            break
        ans = ex["answer"]
        if not ans["value"].strip():
            continue
        rows.append({"id": f"tq-{len(rows):05d}", "family": "triviaqa",
                     "subtype": "closedbook", "template": "closedbook",
                     "question": ex["question"],
                     "answers": list({ans["value"], *ans.get("aliases", [])})})
    return rows


def build_false_premise(n, seed=0):
    """The v1 untemplated-comparison probes, kept so §7-§9 stay reproducible."""
    from build_prompts import build_false_premise as v1
    rows = v1(n, seed)
    for r in rows:
        r["template"] = r["subtype"]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trivia", type=int, default=20000)
    ap.add_argument("--n-template-real", type=int, default=5000)
    ap.add_argument("--n-template-fake", type=int, default=5000)
    ap.add_argument("--n-false", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="prompts_v2.jsonl")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = (build_triviaqa(args.n_trivia, args.seed)
            + build_template_pairs(args.n_template_real, args.n_template_fake,
                                   args.seed)
            + build_false_premise(args.n_false, args.seed))

    path = OUT / args.out
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    c = Counter(r["family"] for r in rows)
    print(f"wrote {len(rows)} prompts -> {path}")
    for k, v in sorted(c.items()):
        print(f"  {k:<15} {v:>6}")
    print("\ntemplate families (real / fake):")
    for t in TEMPLATES:
        r = sum(x["template"] == t and x["family"] == "template_real" for x in rows)
        f = sum(x["template"] == t and x["family"] == "template_fake" for x in rows)
        print(f"  {t:<10} {r:>5} / {f:>5}")
    print("\nexamples:")
    for fam in ("template_real", "template_fake"):
        ex = next(x for x in rows if x["family"] == fam)
        print(f"  [{fam}] {ex['question']}  -> {ex['answers'][:2]}")


if __name__ == "__main__":
    main()
