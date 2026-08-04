"""Command-line entry points. One module per Makefile target.

Every module here is a thin argparse shell that calls into a library and does
nothing else. The rule the layout encodes: **libraries never have ``__main__``,
CLI modules never have logic.** That way an entry point cannot quietly grow
behaviour that the rest of the codebase can't reach or test — which is how
``predict_upcoming.py`` ended up importing a module that no longer existed
without anything noticing.

Invoke as ``uv run python -m src.cli.<name>``; the Makefile does exactly that.
"""
