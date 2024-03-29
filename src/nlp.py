import re
from typing import Generator, List, Sequence, Tuple

import nltk.tag
import nltk.tokenize


def ngrams(tokens: Sequence[str], n: int) -> Generator[Tuple[str, ...], None, None]:
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


TOKENS_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+")


def tokenize(text: str) -> List[str]:
    return TOKENS_RE.findall(text)


LEMMA_RE = re.compile(r"\w+")


def lemmatize(token: str) -> str:
    slices = LEMMA_RE.findall(token)
    return sorted(slices, key=len, reverse=True)[0].lower() if slices else token


def edit_distance(a: str, b: str) -> int:
    if len(b) == 0:
        return len(a)
    elif len(a) == 0:
        return len(b)
    elif a[0] == b[0]:
        return edit_distance(a[1:], b[1:])
    else:
        return 1 + min(edit_distance(a[1:], b), edit_distance(a, b[1:]), edit_distance(a[1:], b[1:]))


def pos_tags(tokens: Sequence[str]) -> List[Tuple[str, str]]:
    return nltk.tag.pos_tag(tokens)


def simplify_tag(tag: str) -> str:
    return nltk.tag.map_tag("en-ptb", "universal", tag)


__all__ = ["ngrams", "tokenize", "lemmatize", "edit_distance", "pos_tags", "simplify_tag"]
