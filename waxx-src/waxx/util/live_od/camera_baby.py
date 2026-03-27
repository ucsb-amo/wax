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
    "Wisp", "Yarrow", "Zephyr", "Zinnia", "Acorn", "Aero", "Aloe", "Alpine",
    "Anchor", "Anvil", "Aqua", "Atlas", "Bamboo", "Beacon", "Biscotti", "Blizzard",
    "Bonfire", "Bonsai", "Brass", "Briar", "Calypso", "Canary", "Canyon", "Cascade",
    "Champ", "Chai", "Cinnamon", "Circuit", "Cliff", "Cobble", "Compass", "Coriander",
    "Crescent", "Crown", "Current", "Delta", "Dewdrop", "Diesel", "Doodle", "Dune",
    "Edison", "Element", "Elm", "Falcon", "Fiddle", "Firefly", "Fjord", "Flare",
    "Floss", "Fountain", "Fractal", "Galaxy", "Garland", "Geode", "Gizmo", "Glacier",
    "Granite", "Groove", "Grove", "Halo", "Hearth", "Helix", "Hickory", "Honeycomb",
    "Hydra", "Indigo", "Ivy", "Jet", "Jigsaw", "Jubilee", "Kite", "Lagoon",
    "Lantern", "Lattice", "Locket", "Magnet", "Marigold", "Meridian", "Meteor", "Midnight",
    "Monsoon", "Mural", "Nugget", "Oak", "Ocean", "Octave", "Orchid", "Parsnip",
    "Peak", "Pecan", "Pioneer", "Plasma", "Prairie", "Prism", "Pulse", "Quasar",
    "Quince", "Radar", "Raven", "Reef", "Ridge", "Rocket", "Rust", "Sable",
    "Saturn", "Scarf", "Sequoia", "Signal", "Slate", "Solstice", "Spindle", "Static",
    "Sunbeam", "Tango", "Tempest", "Terra", "Timber", "Torch", "Tundra", "Umber",
    "Vale", "Vapor", "Verdant", "Vortex", "Walnut", "Whimsy", "Winter", "Wren",
    "Yonder", "Yuzu", "Zenith",
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
    "anchor", "antenna", "applet", "archive", "aster", "atlas", "aurora", "axis",
    "beacon", "binary", "bottle", "breeze", "bridge", "bubble", "cabin", "cadence",
    "canvas", "capsule", "carousel", "cinder", "circuit", "clasp", "coil", "compass",
    "crystal", "cubby", "current", "daisy", "delta", "doodle", "ember", "engine",
    "falcon", "feather", "fig", "fjord", "flame", "flicker", "gadget", "garden",
    "glacier", "glimmer", "groove", "harbor", "harmony", "helix", "hollow", "honey",
    "isotope", "jigsaw", "jolt", "journey", "kernel", "kettle", "lattice", "legend",
    "lily", "lumen", "matrix", "meadow", "melody", "meteor", "mirage", "mosaic",
    "nebula", "needle", "nickel", "oak", "odyssey", "opal", "orchid", "paddle",
    "pantry", "parade", "pearl", "petunia", "photon", "pioneer", "plume", "prairie",
    "pulse", "quartz", "quiver", "radar", "rainbow", "raven", "reef", "relay",
    "ribbon", "ridge", "robin", "saffron", "satellite", "sequoia", "signal", "sonata",
    "spectrum", "sprinkle", "sprocket", "summit", "sunset", "teacup", "thunder", "tinker",
    "topiary", "turbo", "valley", "velour", "vibe", "violet", "vortex", "waffle",
    "whim", "widget", "wildflower", "willow", "winter", "yarn", "yodel", "zen",
]

_HUMAN_NAMES = [
    "Aaliyah", "Aaron", "Abigail", "Adam", "Addison", "Adrian", "Aiden", "Aisha",
    "Alan", "Albert", "Alejandro", "Alexa", "Alexander", "Alexis", "Alice", "Alicia",
    "Allison", "Alyssa", "Amanda", "Amber", "Amelia", "Amir", "Amy", "Andrea",
    "Andrew", "Angela", "Anna", "Anthony", "Aria", "Arianna", "Arthur", "Ashley",
    "Ashton", "Aubrey", "Audrey", "Austin", "Ava", "Axel", "Bailey", "Barbara",
    "Beatrice", "Benjamin", "Bennett", "Bianca", "Blake", "Brandon", "Brayden", "Brenda",
    "Brianna", "Brittany", "Brooke", "Bryan", "Caleb", "Cameron", "Camila", "Carlos",
    "Caroline", "Carter", "Cassandra", "Catherine", "Cecilia", "Chad", "Charles", "Charlotte",
    "Chloe", "Christian", "Christopher", "Claire", "Clara", "Cole", "Colin", "Connor",
    "Courtney", "Crystal", "Daniel", "Danielle", "David", "Dawn", "Dean", "Delilah",
    "Derek", "Diana", "Dominic", "Dylan", "Eleanor", "Elena", "Eli", "Elijah",
    "Elise", "Elizabeth", "Ella", "Ellie", "Emily", "Emma", "Eric", "Erica",
    "Ethan", "Eva", "Evelyn", "Faith", "Felix", "Fiona", "Gabriel", "Gabriella",
    "Gavin", "Gemma", "George", "Gianna", "Grace", "Graham", "Grant", "Hailey",
    "Hannah", "Harper", "Hazel", "Heather", "Henry", "Hudson", "Hunter", "Ian",
    "Isaac", "Isabel", "Isabella", "Isaiah", "Jack", "Jackson", "Jacob", "Jade",
    "James", "Jamie", "Jasmine", "Jason", "Javier", "Jayden", "Jean", "Jenna",
    "Jennifer", "Jeremiah", "Jeremy", "Jessica", "Jillian", "Joanna", "John", "Jonathan",
    "Jordan", "Joseph", "Joshua", "Josiah", "Julia", "Julian", "Julie", "Justin",
    "Kaitlyn", "Kara", "Karen", "Katherine", "Kayla", "Keira", "Kelly", "Kenneth",
    "Kevin", "Kiara", "Kimberly", "Kyle", "Laila", "Lauren", "Layla", "Leah",
    "Leo", "Leon", "Levi", "Liam", "Lillian", "Lily", "Logan", "Lucas",
    "Lucy", "Luis", "Luke", "Luna", "Lydia", "Mackenzie", "Madeline", "Madison",
    "Marcus", "Maria", "Mason", "Matthew", "Maya", "Megan", "Melanie", "Mia",
    "Micah", "Michael", "Michelle", "Mila", "Molly", "Morgan", "Naomi", "Natalie",
    "Nathan", "Nora", "Noah", "Nolan", "Nova", "Oliver", "Olivia", "Omar",
    "Oscar", "Owen", "Paige", "Parker", "Patrick", "Paul", "Penelope", "Peyton",
    "Philip", "Piper", "Quentin", "Rachel", "Rebecca", "Riley", "Robert", "Ryan",
    "Sabrina", "Samantha", "Samuel", "Sara", "Sarah", "Savannah", "Scarlett", "Sean",
    "Sebastian", "Serena", "Seth", "Sienna", "Sofia", "Sophia", "Spencer", "Stella",
    "Stephen", "Steven", "Summer", "Sydney", "Taylor", "Thomas", "Tristan", "Tyler",
    "Valerie", "Vanessa", "Victoria", "Violet", "Vivian", "Walter", "William", "Wyatt",
    "Xavier", "Yasmin", "Zachary", "Zoe", "Zoey",
]

_BIRTH_VERBS = [
    "arrived", "appeared", "blipped into being", "bloomed", "booted up",
    "bounded into service", "bubbled up", "came online", "clocked in",
    "drifted into existence", "emerged", "flickered awake", "floated in",
    "hatched", "materialized", "popped into action", "rolled off the bench",
    "sauntered into the rack", "sparked to life", "sprang forth", "burst out laughing",
    "checked in", "woke up cheerful", "entered stage left", "arrived fashionably late",
    "showed up curious", "came in calibrated", "stood at attention", "appeared with fanfare",
    "clicked into place", "strolled in confidently", "materialized politely",
]

_BIRTH_PLACES = [
    "in the trigger lane", "at the command port", "by the camera rack",
    "near the viewport", "under the lab lights", "inside the grab loop",
    "beside the fiber spool", "in the control room", "on the data rail",
    "at the experiment bench", "in the timing chain", "near the sync line",
    "by the patch panel", "in front of the monitor wall", "by the cooling loop",
    "next to the shutter stack", "on the acquisition node", "at the rack door",
    "between the trigger edges",
]

_HONORABLE_DEATH_VERBS = [
    "clocked out", "drifted off", "ended the shift", "finished the tour",
    "folded up the stroller", "hung up the pacifier", "left the nursery",
    "packed up the bottle", "powered down", "sailed into retirement",
    "slipped into downtime", "stepped off stage", "turned in the badge",
    "wrapped the shift", "went off duty", "took a graceful bow",
    "called it a good day", "retired in style", "signed off proudly",
    "closed the notebook", "powered down with dignity", "took a deserved nap",
    "left to thunderous applause",
]

_HONORABLE_DEATH_PLACES = [
    "after a clean run", "by the viewer window", "with the shutters quiet",
    "under the rack fans", "without a fuss", "with excellent timing",
    "at the end of the sequence", "under calm skies", "with the buffers empty",
    "after the last image landed", "with every frame accounted for",
    "after a textbook run", "with the timeline complete", "while status lights smiled",
    "as the queue hit zero", "with all checks green", "after perfect synchronization",
    "as the final ack returned",
]

_DISHONORABLE_DEATH_VERBS = [
    "got benched", "got bounced", "got called home early", "hit a snag",
    "ran out of runway", "stumbled out", "timed out", "took a spill",
    "was escorted out", "was pulled from duty", "was sent packing",
    "was shuffled off", "went sideways", "wiped out", "got unplugged",
    "hit the emergency stop", "lost the plot", "tripped over a cable",
    "got zapped by gremlins", "faceplanted in the queue", "missed the handshake",
    "fell into a timeout pit", "crashed out dramatically",
]

_DISHONORABLE_DEATH_PLACES = [
    "after a rough reset", "before the last trigger", "by the fault light",
    "in the middle of the sequence", "under less-than-ideal conditions",
    "with buffers still warm", "before the shutters settled",
    "while the rack was grumpy", "with the viewers still waiting",
    "before the finish line", "while packets were still in flight",
    "during an awkward reconnect", "with a suspicious stack trace",
    "while the watchdog frowned", "midway through a burst", "before the final ack",
    "right after saying it was fine", "in a cloud of confusion",
]


def _combine(parts_a: list[str], parts_b: list[str]) -> list[str]:
    return [f"{a}{b}" for a in parts_a for b in parts_b]


def _build_camera_baby_names() -> list[str]:
    """Create a mixed name pool of human names and prefix/suffix names."""
    combo_names = _combine(_PREFIXES, _SUFFIXES)

    mixed: list[str] = []
    max_len = max(len(_HUMAN_NAMES), len(combo_names))
    for i in range(max_len):
        if i < len(_HUMAN_NAMES):
            mixed.append(_HUMAN_NAMES[i])
        if i < len(combo_names):
            mixed.append(combo_names[i])
    return mixed


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


CAMERA_BABY_NAMES = _build_camera_baby_names()
BIRTH_EUPHEMISMS = _build_birth_phrases()
HONORABLE_DEATH_EUPHEMISMS = _build_phrases(
    _HONORABLE_DEATH_VERBS,
    _HONORABLE_DEATH_PLACES,
)
DISHONORABLE_DEATH_EUPHEMISMS = _build_phrases(
    _DISHONORABLE_DEATH_VERBS,
    _DISHONORABLE_DEATH_PLACES,
)


if len(CAMERA_BABY_NAMES) < 1500:
    raise RuntimeError("Expected at least 1500 camera baby names.")
if len(BIRTH_EUPHEMISMS) < 150:
    raise RuntimeError("Expected at least 150 birth euphemisms.")
if len(HONORABLE_DEATH_EUPHEMISMS) < 150:
    raise RuntimeError("Expected at least 150 honorable death euphemisms.")
if len(DISHONORABLE_DEATH_EUPHEMISMS) < 150:
    raise RuntimeError("Expected at least 150 dishonorable death euphemisms.")
