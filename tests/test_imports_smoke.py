"""Every module must import.

This test exists because it was absent. ``src/ml/scripts/predict_upcoming.py``
imported a module that had been deleted, and ``predict_classifier.py`` imported
a name from another script that no longer re-exported it — so ``make
predict-upcoming``, the project's core inference path, raised ImportError. The
unit tests all passed the whole time, because nothing imported those entry
points.

Importing is the cheapest possible check and catches the entire class of "a
rename left a caller behind".
"""

import importlib
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"


def _module_names() -> list[str]:
    return sorted(
        str(path.relative_to(SRC.parent).with_suffix("")).replace("/", ".")
        for path in SRC.rglob("*.py")
        if "__pycache__" not in path.parts and path.name != "__init__.py"
    )


@pytest.mark.parametrize("module_name", _module_names())
def test_module_imports(module_name):
    importlib.import_module(module_name)


def test_every_cli_module_has_a_main():
    """Each CLI module is an entry point; the Makefile invokes it via -m."""
    for module_name in _module_names():
        if not module_name.startswith("src.cli."):
            continue
        module = importlib.import_module(module_name)
        assert callable(getattr(module, "main", None)), f"{module_name} has no main()"


def test_only_cli_modules_are_executable():
    """Libraries must not carry ``__main__`` blocks.

    Entry points belong in src/cli/ so behaviour cannot hide in a script that
    nothing imports — which is exactly how the broken paths above went unnoticed.
    """
    offenders = [
        str(path.relative_to(SRC.parent))
        for path in SRC.rglob("*.py")
        if "__pycache__" not in path.parts
        and "cli" not in path.parts
        and '__name__ == "__main__"' in path.read_text()
    ]
    assert not offenders, f"__main__ block outside src/cli/: {offenders}"
