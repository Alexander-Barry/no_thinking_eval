"""Shallow heuristics for iterate_map36, the letters family over the 36 symbols A-Z and 0-9.

The rules have the same shape as the 26-letter family ("g(A) = 7, ...", "exactly k times"), and the listing order
is A-Z then 0-9, so the letters battery applies unchanged; this module reuses it under the new family name.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

FAMILY = "iterate_map36"

_spec = importlib.util.spec_from_file_location("heur_iterate_map", Path(__file__).with_name("heur_iterate_map.py"))
_letters = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_letters)


def heuristics() -> dict:
    return _letters.heuristics()
