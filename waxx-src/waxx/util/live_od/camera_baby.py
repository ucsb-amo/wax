"""Camera baby naming and phrase helpers.

Keeps the novelty strings out of ``camera_server.py`` and guarantees a large
pool of unique names / euphemisms so the logs do not repeat too quickly.
"""

from __future__ import annotations

_PREFIXES = [
    "Amber", "Apple", "Apricot", "Arbor", "Ash", "Aster", "Aurora", "Autumn",
    "Basil", "Bay", "Berry", "Birch", "Blossom", "Blue", "Bramble", "Breeze",
    "Brook", "Butter", "Cactus", "Caper", "Cedar", "Cherry", "Cinder", "Citrus",
    "Clover", "Cloud", "Cobalt", "Coco", "Comet", "Copper", "Coral", "Cosmo",
    "Cricket", "Crisp", "Daisy", "Dandelion", "Dawn", "Drift", "Echo", "Ember",
    "Fable", "Fern", "Fizz", "Flint", "Flora", "Fog", "Forest", "Frost",
    "Gadget", "Gale", "Ginger", "Glimmer", "Glow", "Gossamer", "Harbor", "Hazel",
    "Honey", "Horizon", "Iris", "Ivory", "Jade", "Jasper", "Jelly", "Juniper",
    "Kelp", "Kindle", "Kiwi", "Lake", "Lavender", "Lemon", "Lilac", "Lime",
    "Linen", "Lotus", "Luna", "Maple", "Marble", "Meadow", "Mica", "Mint",
    "Miso", "Mist", "Moon", "Moss", "Nectar", "Needle", "Nimbus", "Nova",
    "Olive", "Onyx", "Opal", "Orbit", "Pebble", "Pepper", "Petal", "Pine",
    "Pixel", "Plum", "Pollen", "Poppy", "Quartz", "Rain", "River", "Robin",
    "Rose", "Saffron", "Sage", "Shadow", "Silk", "Silver", "Sky", "Smoke",
    "Solar", "Sorrel", "Spark", "Sprout", "Star", "Stone", "Storm", "Summer",
    "Sunny", "Thistle", "Thyme", "Toffee", "Topaz", "Tulip", "Velvet", "Willow",
    "Wisp", "Yarrow", "Zephyr", "Zinnia",
]

_SUFFIXES = [
    "beam", "bell", "berry", "biscuit", "blink", "blip", "bloom", "bud",
    "button", "cake", "chip", "chirp", "cloud", "clover", "comet", "cookie",
    "cricket", "crumb", "cube", "cup", "dash", "dew", "dot", "dream",
    "drop", "drum", "dust", "fern", "fizz", "flake", "flash", "flower",
    "flutter", "foam", "frond", "gem", "glow", "hopper", "jelly", "joy",
    "leaf", "light", "loop", "marble", "midge", "muffin", "nest", "noodle",
    "nova", "orbit", "patch", "pebble", "petal", "pickle", "pocket", "puff",
    "quill", "ripple", "rocket", "seed", "shine", "sketch", "sprig", "sprout",
    "star", "stone", "stripe", "sugar", "swirl", "thimble", "thread", "toast",
    "toffee", "twig", "velvet", "whistle", "wink", "wisp", "zest", "zip",
]

_BIRTH_VERBS = [
    "arrived", "appeared", "blipped into being", "bloomed", "booted up",
    "bounded into service", "bubbled up", "came online", "clocked in",
    "drifted into existence", "emerged", "flickered awake", "floated in",
    "hatched", "materialized", "popped into action", "rolled off the bench",
    "sauntered into the rack", "sparked to life", "sprang forth",
]

_BIRTH_PLACES = [
    "in the trigger lane", "at the command port", "by the camera rack",
    "near the viewport", "under the lab lights", "inside the grab loop",
    "beside the fiber spool", "in the control room", "on the data rail",
    "at the experiment bench",
]

_HONORABLE_DEATH_VERBS = [
    "clocked out", "drifted off", "ended the shift", "finished the tour",
    "folded up the stroller", "hung up the pacifier", "left the nursery",
    "packed up the bottle", "powered down", "sailed into retirement",
    "slipped into downtime", "stepped off stage", "turned in the badge",
    "wrapped the shift", "went off duty",
]

_HONORABLE_DEATH_PLACES = [
    "after a clean run", "by the viewer window", "with the shutters quiet",
    "under the rack fans", "without a fuss", "with excellent timing",
    "at the end of the sequence", "under calm skies", "with the buffers empty",
    "after the last image landed",
]

_DISHONORABLE_DEATH_VERBS = [
    "got benched", "got bounced", "got called home early", "hit a snag",
    "ran out of runway", "stumbled out", "timed out", "took a spill",
    "was escorted out", "was pulled from duty", "was sent packing",
    "was shuffled off", "went sideways", "wiped out", "got unplugged",
]

_DISHONORABLE_DEATH_PLACES = [
    "after a rough reset", "before the last trigger", "by the fault light",
    "in the middle of the sequence", "under less-than-ideal conditions",
    "with buffers still warm", "before the shutters settled",
    "while the rack was grumpy", "with the viewers still waiting",
    "before the finish line",
]


def _combine(parts_a: list[str], parts_b: list[str]) -> list[str]:
    return [f"{a}{b}" for a in parts_a for b in parts_b]


def _build_birth_phrases() -> list[str]:
    phrases: list[str] = []
    for verb in _BIRTH_VERBS:
        for place in _BIRTH_PLACES:
            phrases.append(f"{verb} {place}")
    return phrases


def _build_phrases(verbs: list[str], places: list[str]) -> list[str]:
    phrases: list[str] = []
    for verb in verbs:
        for place in places:
            phrases.append(f"{verb} {place}")
    return phrases


CAMERA_BABY_NAMES = _combine(_PREFIXES, _SUFFIXES)
BIRTH_EUPHEMISMS = _build_birth_phrases()
HONORABLE_DEATH_EUPHEMISMS = _build_phrases(
    _HONORABLE_DEATH_VERBS,
    _HONORABLE_DEATH_PLACES,
)
DISHONORABLE_DEATH_EUPHEMISMS = _build_phrases(
    _DISHONORABLE_DEATH_VERBS,
    _DISHONORABLE_DEATH_PLACES,
)


if len(CAMERA_BABY_NAMES) < 1000:
    raise RuntimeError("Expected at least 1000 camera baby names.")
if len(BIRTH_EUPHEMISMS) < 75:
    raise RuntimeError("Expected at least 75 birth euphemisms.")
if len(HONORABLE_DEATH_EUPHEMISMS) < 75:
    raise RuntimeError("Expected at least 75 honorable death euphemisms.")
if len(DISHONORABLE_DEATH_EUPHEMISMS) < 75:
    raise RuntimeError("Expected at least 75 dishonorable death euphemisms.")
