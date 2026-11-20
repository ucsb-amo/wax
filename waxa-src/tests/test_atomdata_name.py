"""``waxa.atomdata`` is the class whichever waxa module loaded first. Each case runs in a fresh
interpreter, because the bug depends on import order."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[1])


def run(code):
    env = dict(os.environ, PYTHONPATH=SRC)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)


@pytest.mark.parametrize("first", [
    "from waxa import atomdata",
    "from waxa import AtomdataVault",
    "import waxa.atomdata",
    "import waxa.atomdata_vault",
    "from waxa.atomdata import unpack_group",
])
def test_the_name_is_the_class_after_any_first_import(first):
    result = run(f"{first}\nfrom waxa import atomdata\nassert isinstance(atomdata, type), atomdata")
    assert result.returncode == 0, result.stderr


def test_names_inside_the_submodule_still_import():
    result = run("from waxa import AtomdataVault\nfrom waxa.atomdata import atomdata, unpack_group\n"
                 "import sys\nassert type(sys.modules['waxa.atomdata']).__name__ == 'module'\n"
                 "import waxa\nassert waxa.atomdata is atomdata")
    assert result.returncode == 0, result.stderr


def test_importing_waxa_stays_light():
    """The lazy imports exist so that loky workers do not load server_talk or PyQt6."""
    result = run("import sys, waxa\nassert 'waxa.data.server_talk' not in sys.modules\nassert 'PyQt6' not in sys.modules")
    assert result.returncode == 0, result.stderr
