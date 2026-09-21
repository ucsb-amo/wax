"""Expt._expt_file_stem must name the experiment file when ARTIQ loads it.

ARTIQ's file_import does not register the module in sys.modules, so
inspect.getsourcefile(cls) fails there and liveOD fell back to the class name.
"""
from artiq.tools import file_import

from waxx.base.expt import Expt


def _load(tmp_path, stem, body):
    path = tmp_path / f"{stem}.py"
    path.write_text(body)
    return file_import(str(path))


def test_stem_from_artiq_file_import(tmp_path):
    mod = _load(tmp_path, "hf_foo_evap",
                "from artiq.experiment import kernel\n"
                "class HFFoo:\n"
                "    @kernel\n"
                "    def run(self):\n"
                "        pass\n")
    assert Expt._expt_file_stem(mod.HFFoo()) == "hf_foo_evap"


def test_stem_falls_back_to_module_name(tmp_path):
    mod = _load(tmp_path, "no_methods", "class Bare:\n    pass\n")
    assert Expt._expt_file_stem(mod.Bare()) == "no_methods"


def test_stem_for_normally_imported_class():
    assert Expt._expt_file_stem(Expt.__new__(Expt)) == "expt"
