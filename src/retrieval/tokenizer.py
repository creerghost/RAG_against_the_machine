"""Identifier-aware tokenizer shared by indexing and querying."""

import re
from typing import Iterable


class Tokenizer:
    """Lowercased words plus snake_case and CamelCase parts of identifiers."""
    word_regex = re.compile(r"\w+")
    camel_regex = re.compile(
        r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

    def __init__(self, stopwords: Iterable[str], min_length: int) -> None:
        """Drop ``stopwords`` and tokens shorter than ``min_length`` chars."""
        self.stopwords = frozenset(w.lower() for w in stopwords)
        self.min_length = min_length

    def tokenize(self, text: str) -> list[str]:
        """Return ``text``'s tokens in order: identifier parts, then whole."""
        result: list[str] = []
        for match in self.word_regex.finditer(text):
            token = match.group()
            parts = [c for p in token.split("_")
                     for c in self.camel_regex.split(p) if c]
            if parts != [token]:  # also "__init__" -> "init"
                result.extend(p.lower() for p in parts)
            result.append(token.lower())
        return [t for t in result if self._keep(t)]

    def _keep(self, token: str) -> bool:
        """Return False for stopwords, short tokens and tokens like ``___``."""
        return (len(token) >= self.min_length
                and token not in self.stopwords
                and any(c.isalnum() for c in token))
