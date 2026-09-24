"""Fuzzy name matching shared by the data browser and the device control GUI.

Pure functions, no Qt and no plotting imports, so a GUI that only wants
``parse_name_search_terms`` / ``name_matches_all_terms`` does not pay for
matplotlib and scipy the way it did when these lived in
``waxa.browser.browser_window`` (about 1.1 s at import).  That module still
re-exports them, so existing imports keep working.

Search syntax: terms are separated by ``+`` and every term must match.  A
term matches a name when it is a substring of the raw lowercase name, a
substring of the name with non-alphanumerics stripped, or an in-order
subsequence of that stripped name (so ``ttlcam`` matches ``ttl_camera``).
"""

from __future__ import annotations

import functools
import re


def parse_name_search_terms(query: str) -> list[str]:
    normalized_query = (query or "").strip().lower()
    return [term.strip() for term in normalized_query.split("+") if term.strip()]


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]")


@functools.lru_cache(maxsize=65536)
def normalize_match_text(value: str) -> str:
    # Memoised: the same xvar/experiment names recur across thousands of runs,
    # and this runs once per name per filter term on every filter pass.
    return _NON_ALNUM_RE.sub("", (value or "").lower())


def is_subsequence(needle: str, haystack: str) -> bool:
    if not needle:
        return True
    index = 0
    for char in haystack:
        if char == needle[index]:
            index += 1
            if index == len(needle):
                return True
    return False


def name_matches_term(term: str, raw_name: str) -> bool:
    raw_name = (raw_name or "").lower()
    normalized_name = normalize_match_text(raw_name)
    normalized_term = normalize_match_text(term)
    if term in raw_name:
        return True
    if normalized_term and normalized_term in normalized_name:
        return True
    if normalized_term and is_subsequence(normalized_term, normalized_name):
        return True
    return False


def name_matches_all_terms(raw_name: str, terms: list[str]) -> bool:
    if not terms:
        return True
    return all(name_matches_term(term, raw_name) for term in terms)


def any_name_matches_all_terms(names: list[str], terms: list[str]) -> bool:
    if not terms:
        return True
    lowered_names = [str(name).lower() for name in names]
    for term in terms:
        if not any(name_matches_term(term, name) for name in lowered_names):
            return False
    return True


__all__ = [
    "parse_name_search_terms",
    "normalize_match_text",
    "is_subsequence",
    "name_matches_term",
    "name_matches_all_terms",
    "any_name_matches_all_terms",
]
