import re
from typing import Generator, List, Sequence, Tuple


def ngrams(tokens: Sequence[str], n: int) -> Generator[Tuple[str, ...], None, None]:
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def tokenize(text: str) -> List[str]:
    text = re.sub(r"[^\w\s]", " ", text)

    return text.split()
