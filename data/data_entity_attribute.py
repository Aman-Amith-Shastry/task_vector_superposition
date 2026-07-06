"""
Data generation for the semantic-relation entity→attribute experiment.

Three tasks share an identical surface form — a country name in, a short factual
answer out — but differ in the SEMANTIC RELATION the in-context demonstrations
teach:

  Capital   "France"  ->  "Paris"
  Currency  "France"  ->  "Euro"
  Language  "France"  ->  "French"

Two properties make this group the right vehicle for *behavioral* injection,
unlike the format-varying arithmetic group:

  1. The task is signalled ONLY by the in-context demonstrations, never by the
     prompt format — the test question is a bare country name. So the contrast
     vector captures the task identity, and the zero-shot baseline ("Brazil")
     is genuinely task-neutral.
  2. Every task shares the same output affordance: the same prompt "Brazil" can
     be answered as a capital, a currency, or a language. There is no scaffolding
     asymmetry (the arithmetic MCQ task needed options on screen; nothing here
     does), so injecting any pure vector asks for something the prompt can express.

Multi-token answers ("New Delhi", "Buenos Aires") are fine: the behavioral
readout uses teacher-forced scoring of the gold answer (score_continuation),
not single first-token logits.

NOTE on task choice — we use Capital / Currency / Language rather than
Capital / City / Language. "City" (largest city) collapses into "Capital" for
most countries (Paris, Tokyo, London are both), which makes those two tasks
non-distinct, degenerates the superposition simplex, and shrinks the country
pool to the ~20 countries where capital != largest city. Currency avoids all of
this. To switch to largest-city anyway: rename the "currency" field below to
"city", fill it with each country's largest city, restrict COUNTRIES to those
where capital != city, and update TASKS / _ATTR accordingly.

Currency and language are given as their most common single name (e.g. "Euro",
"Pound", "Spanish"); answer collisions across countries (many "Euro"/"Peso"
currencies) are harmless — the task is the *relation*, not answer diversity.
"""

import random

TASKS            = ["Capital", "Currency", "Language"]
N_VECTOR_SAMPLES = 20
SEED             = 42

# Country attribute -> task mapping
_ATTR = {"Capital": "capital", "Currency": "currency", "Language": "language"}

# (n_capital, n_currency, n_language) — must sum to 3
RATIOS: list[tuple[int, int, int]] = [
    (3, 0, 0),
    (0, 3, 0),
    (0, 0, 3),
    (2, 1, 0),
    (1, 2, 0),
    (0, 2, 1),
    (0, 1, 2),
    (2, 0, 1),
    (1, 0, 2),
    (1, 1, 1),
]
PURE_RATIOS: list[tuple[int, int, int]] = [(3, 0, 0), (0, 3, 0), (0, 0, 3)]

# How many countries are held out for the test pool (disjoint from ICL pool, so
# a prompt's test country never appears among its own demonstrations). The test
# pool is what the behavioral injection averages over, so a larger pool tightens
# the bootstrap CIs on the steering effect.
N_TEST_COUNTRIES = 30

# country, capital, currency, language  (most-common single name for each)
COUNTRIES: list[dict] = [
    {"country": "France",       "capital": "Paris",        "currency": "Euro",     "language": "French"},
    {"country": "Japan",        "capital": "Tokyo",        "currency": "Yen",      "language": "Japanese"},
    {"country": "Germany",      "capital": "Berlin",       "currency": "Euro",     "language": "German"},
    {"country": "Italy",        "capital": "Rome",         "currency": "Euro",     "language": "Italian"},
    {"country": "Spain",        "capital": "Madrid",       "currency": "Euro",     "language": "Spanish"},
    {"country": "China",        "capital": "Beijing",      "currency": "Yuan",     "language": "Mandarin"},
    {"country": "Russia",       "capital": "Moscow",       "currency": "Ruble",    "language": "Russian"},
    {"country": "Brazil",       "capital": "Brasilia",     "currency": "Real",     "language": "Portuguese"},
    {"country": "India",        "capital": "Delhi",        "currency": "Rupee",    "language": "Hindi"},
    {"country": "Canada",       "capital": "Ottawa",       "currency": "Dollar",   "language": "English"},
    {"country": "Australia",    "capital": "Canberra",     "currency": "Dollar",   "language": "English"},
    {"country": "Egypt",        "capital": "Cairo",        "currency": "Pound",    "language": "Arabic"},
    {"country": "Turkey",       "capital": "Ankara",       "currency": "Lira",     "language": "Turkish"},
    {"country": "Greece",       "capital": "Athens",       "currency": "Euro",     "language": "Greek"},
    {"country": "Portugal",     "capital": "Lisbon",       "currency": "Euro",     "language": "Portuguese"},
    {"country": "Netherlands",  "capital": "Amsterdam",    "currency": "Euro",     "language": "Dutch"},
    {"country": "Sweden",       "capital": "Stockholm",    "currency": "Krona",    "language": "Swedish"},
    {"country": "Norway",       "capital": "Oslo",         "currency": "Krone",    "language": "Norwegian"},
    {"country": "Poland",       "capital": "Warsaw",       "currency": "Zloty",    "language": "Polish"},
    {"country": "Thailand",     "capital": "Bangkok",      "currency": "Baht",     "language": "Thai"},
    {"country": "Vietnam",      "capital": "Hanoi",        "currency": "Dong",     "language": "Vietnamese"},
    {"country": "Indonesia",    "capital": "Jakarta",      "currency": "Rupiah",   "language": "Indonesian"},
    {"country": "Iran",         "capital": "Tehran",       "currency": "Rial",     "language": "Persian"},
    {"country": "Kenya",        "capital": "Nairobi",      "currency": "Shilling", "language": "Swahili"},
    {"country": "Nigeria",      "capital": "Abuja",        "currency": "Naira",    "language": "English"},
    {"country": "Peru",         "capital": "Lima",         "currency": "Sol",      "language": "Spanish"},
    {"country": "Chile",        "capital": "Santiago",     "currency": "Peso",     "language": "Spanish"},
    {"country": "Colombia",     "capital": "Bogota",       "currency": "Peso",     "language": "Spanish"},
    {"country": "Switzerland",  "capital": "Bern",         "currency": "Franc",    "language": "German"},
    {"country": "Austria",      "capital": "Vienna",       "currency": "Euro",     "language": "German"},
    {"country": "Denmark",      "capital": "Copenhagen",   "currency": "Krone",    "language": "Danish"},
    {"country": "Finland",      "capital": "Helsinki",     "currency": "Euro",     "language": "Finnish"},
    {"country": "Ireland",      "capital": "Dublin",       "currency": "Euro",     "language": "English"},
    {"country": "Ukraine",      "capital": "Kyiv",         "currency": "Hryvnia",  "language": "Ukrainian"},
    {"country": "Hungary",      "capital": "Budapest",     "currency": "Forint",   "language": "Hungarian"},
    {"country": "Romania",      "capital": "Bucharest",    "currency": "Leu",      "language": "Romanian"},
    {"country": "South Korea",  "capital": "Seoul",        "currency": "Won",      "language": "Korean"},
    {"country": "Pakistan",     "capital": "Islamabad",    "currency": "Rupee",    "language": "Urdu"},
    {"country": "Bangladesh",   "capital": "Dhaka",        "currency": "Taka",     "language": "Bengali"},
    {"country": "Malaysia",     "capital": "Kuala Lumpur", "currency": "Ringgit",  "language": "Malay"},
    {"country": "Morocco",      "capital": "Rabat",        "currency": "Dirham",   "language": "Arabic"},
    {"country": "Ghana",        "capital": "Accra",        "currency": "Cedi",     "language": "English"},
    {"country": "Cambodia",     "capital": "Phnom Penh",   "currency": "Riel",     "language": "Khmer"},
    {"country": "Iceland",      "capital": "Reykjavik",    "currency": "Krona",    "language": "Icelandic"},
    {"country": "Bulgaria",     "capital": "Sofia",        "currency": "Lev",      "language": "Bulgarian"},
    {"country": "Serbia",       "capital": "Belgrade",     "currency": "Dinar",    "language": "Serbian"},
    {"country": "Kazakhstan",   "capital": "Astana",       "currency": "Tenge",    "language": "Kazakh"},
    {"country": "Mongolia",     "capital": "Ulaanbaatar",  "currency": "Tugrik",   "language": "Mongolian"},
    {"country": "Saudi Arabia", "capital": "Riyadh",       "currency": "Riyal",    "language": "Arabic"},
    {"country": "Argentina",    "capital": "Buenos Aires", "currency": "Peso",     "language": "Spanish"},
    {"country": "Mexico",       "capital": "Mexico City",  "currency": "Peso",     "language": "Spanish"},
    {"country": "Czechia",      "capital": "Prague",       "currency": "Koruna",   "language": "Czech"},
    {"country": "Croatia",      "capital": "Zagreb",       "currency": "Euro",     "language": "Croatian"},
    {"country": "Slovakia",     "capital": "Bratislava",   "currency": "Euro",     "language": "Slovak"},
    {"country": "Slovenia",     "capital": "Ljubljana",    "currency": "Euro",     "language": "Slovene"},
    {"country": "Lithuania",    "capital": "Vilnius",      "currency": "Euro",     "language": "Lithuanian"},
    {"country": "Latvia",       "capital": "Riga",         "currency": "Euro",     "language": "Latvian"},
    {"country": "Estonia",      "capital": "Tallinn",      "currency": "Euro",     "language": "Estonian"},
    {"country": "Philippines",  "capital": "Manila",       "currency": "Peso",     "language": "Filipino"},
    {"country": "New Zealand",  "capital": "Wellington",   "currency": "Dollar",   "language": "English"},
    {"country": "South Africa", "capital": "Pretoria",     "currency": "Rand",     "language": "English"},
    {"country": "Israel",       "capital": "Jerusalem",    "currency": "Shekel",   "language": "Hebrew"},
    {"country": "Iraq",         "capital": "Baghdad",      "currency": "Dinar",    "language": "Arabic"},
    {"country": "Jordan",       "capital": "Amman",        "currency": "Dinar",    "language": "Arabic"},
    {"country": "Lebanon",      "capital": "Beirut",       "currency": "Pound",    "language": "Arabic"},
    {"country": "Qatar",        "capital": "Doha",         "currency": "Riyal",    "language": "Arabic"},
    {"country": "Kuwait",       "capital": "Kuwait City",  "currency": "Dinar",    "language": "Arabic"},
    {"country": "Nepal",        "capital": "Kathmandu",    "currency": "Rupee",    "language": "Nepali"},
    {"country": "Sri Lanka",    "capital": "Colombo",      "currency": "Rupee",    "language": "Sinhala"},
    {"country": "Laos",         "capital": "Vientiane",    "currency": "Kip",      "language": "Lao"},
    {"country": "Afghanistan",  "capital": "Kabul",        "currency": "Afghani",  "language": "Pashto"},
    {"country": "Ethiopia",     "capital": "Addis Ababa",  "currency": "Birr",     "language": "Amharic"},
    {"country": "Tanzania",     "capital": "Dodoma",       "currency": "Shilling", "language": "Swahili"},
    {"country": "Algeria",      "capital": "Algiers",      "currency": "Dinar",    "language": "Arabic"},
    {"country": "Tunisia",      "capital": "Tunis",        "currency": "Dinar",    "language": "Arabic"},
    {"country": "Senegal",      "capital": "Dakar",        "currency": "Franc",    "language": "French"},
    {"country": "Cameroon",     "capital": "Yaounde",      "currency": "Franc",    "language": "French"},
    {"country": "Angola",       "capital": "Luanda",       "currency": "Kwanza",   "language": "Portuguese"},
    {"country": "Mozambique",   "capital": "Maputo",       "currency": "Metical",  "language": "Portuguese"},
    {"country": "Zimbabwe",     "capital": "Harare",       "currency": "Dollar",   "language": "English"},
    {"country": "Zambia",       "capital": "Lusaka",       "currency": "Kwacha",   "language": "English"},
    {"country": "Ecuador",      "capital": "Quito",        "currency": "Dollar",   "language": "Spanish"},
    {"country": "Venezuela",    "capital": "Caracas",      "currency": "Bolivar",  "language": "Spanish"},
    {"country": "Uruguay",      "capital": "Montevideo",   "currency": "Peso",     "language": "Spanish"},
    {"country": "Paraguay",     "capital": "Asuncion",     "currency": "Guarani",  "language": "Spanish"},
    {"country": "Cuba",         "capital": "Havana",       "currency": "Peso",     "language": "Spanish"},
    {"country": "Guatemala",    "capital": "Guatemala City","currency": "Quetzal", "language": "Spanish"},
    {"country": "Costa Rica",   "capital": "San Jose",     "currency": "Colon",    "language": "Spanish"},
    {"country": "Panama",       "capital": "Panama City",  "currency": "Balboa",   "language": "Spanish"},
    {"country": "Dominican Republic", "capital": "Santo Domingo", "currency": "Peso", "language": "Spanish"},
    {"country": "Jamaica",      "capital": "Kingston",     "currency": "Dollar",   "language": "English"},
    {"country": "Belarus",      "capital": "Minsk",        "currency": "Ruble",    "language": "Belarusian"},
    {"country": "Georgia",      "capital": "Tbilisi",      "currency": "Lari",     "language": "Georgian"},
    {"country": "Armenia",      "capital": "Yerevan",      "currency": "Dram",     "language": "Armenian"},
    {"country": "Azerbaijan",   "capital": "Baku",         "currency": "Manat",    "language": "Azerbaijani"},
    {"country": "Uzbekistan",   "capital": "Tashkent",     "currency": "Som",      "language": "Uzbek"},
]


# --------------------------------------------------------------------------
# Pools
# --------------------------------------------------------------------------

def load_pools() -> dict[str, dict]:
    """ICL and test pools for all three relation tasks.

    The country list is split once into disjoint ICL and test sets so a prompt's
    test country can never appear among its own demonstrations. Each pool entry
    is the full country record; the format helpers select the per-task attribute.
    All three tasks share the same countries (only the demonstrated relation
    differs), mirroring how the arithmetic group shares op pairs across formats.
    """
    shuffled = COUNTRIES[:]
    random.Random(SEED).shuffle(shuffled)
    test_countries = shuffled[:N_TEST_COUNTRIES]
    icl_countries  = shuffled[N_TEST_COUNTRIES:]

    pools: dict[str, dict] = {}
    for task in TASKS:
        pools[task] = {
            "icl":  [dict(c) for c in icl_countries],
            "test": [dict(c) for c in test_countries],
        }
    return pools


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

def format_icl_example(task: str, rec: dict) -> tuple[str, str]:
    """(user_content, assistant_content) for one demonstration of `task`."""
    return rec["country"], rec[_ATTR[task]]


def format_test_input(task: str, rec: dict) -> str:
    """User content for the test question — a bare country name.

    Task-independent by design: the relation is signalled only by the ICL
    demonstrations, so the zero-shot baseline is identical across tasks and the
    output affordance is symmetric. `task` is accepted for interface parity with
    the other data modules but is intentionally unused.
    """
    return rec["country"]


def gold_answer(task: str, rec: dict) -> str:
    """The correct answer string for `task` on country record `rec`.

    Used by the behavioral injection readout to teacher-force-score the gold
    continuation per task.
    """
    return rec[_ATTR[task]]


# --------------------------------------------------------------------------
# build_messages
# --------------------------------------------------------------------------

def build_messages(
    pools: dict,
    rng: random.Random,
    counts: tuple[int, int, int],
    sample_idx: int = 0,
    rotate_test: bool = False,
) -> list[dict]:
    """ICL messages for the given per-task counts + a held-out test question.

    rng call order matches data_arithmetic_formats.build_messages exactly (test
    selection first, then per-task ICL sampling, then shuffle) so a sweep script
    copied from sweep_arithmetic.py reconstructs the zero-shot test prompt
    correctly from the same seed.

    With rotate_test=True the test country rotates by sample_idx; since the test
    input is just a country name, this balances which country contributes rather
    than which format (there is no format to balance here).
    """
    is_pure = counts in PURE_RATIOS
    if rotate_test:
        test_task = TASKS[sample_idx % len(TASKS)]
    else:
        test_task = TASKS[counts.index(max(counts))] if is_pure else TASKS[sample_idx % len(TASKS)]

    test_rec   = rng.choice(pools[test_task]["test"])
    test_input = format_test_input(test_task, test_rec)

    all_examples: list[tuple[str, str]] = []
    for task, n in zip(TASKS, counts):
        for rec in rng.sample(pools[task]["icl"], n):
            all_examples.append(format_icl_example(task, rec))
    rng.shuffle(all_examples)

    messages: list[dict] = []
    for inp, out in all_examples:
        messages.append({"role": "user",      "content": inp})
        messages.append({"role": "assistant", "content": out})
    messages.append({"role": "user", "content": test_input})
    return messages
