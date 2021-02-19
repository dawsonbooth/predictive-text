import re
from typing import Generator, List, Sequence, Tuple


def ngrams(tokens: Sequence[str], n: int) -> Generator[Tuple[str, ...], None, None]:
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


token_re = re.compile(r"[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*")


def tokenize(text: str) -> List[str]:
    return token_re.findall(text)
