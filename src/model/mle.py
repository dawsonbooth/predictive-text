import enum
import math
from typing import Dict, List, Optional, Tuple

from nlp import ngrams, tokenize

from .model import Model


class Smoothing(enum.Enum):
    NONE = enum.auto()
    LAPLACE = enum.auto()
    GOOD_TURING = enum.auto()


class MLE(Model):
    __slots__ = "n", "smoothing", "history", "tokens", "unigram_counts", "ngram_counts"

    n: int
    smoothing: Smoothing
    history: int

    tokens: List[str]
    unigram_counts: Dict[str, int]
    ngram_counts: Dict[Tuple[str, ...], int]

    def __init__(self, n: int = 2, smoothing: Optional[Smoothing] = None, history: int = -1) -> None:
        super().__init__()
        self.n = n
        self.smoothing = smoothing or Smoothing.NONE
        self.history = max(history, self.n)

    def fit(self, text: str) -> None:
        self.tokens = tokenize(text)

        self.unigram_counts = dict()
        for unigram in self.tokens:
            self.unigram_counts[unigram] = self.unigram_counts.get(unigram, 0) + 1

        self.ngram_counts = dict()
        for ngram in ngrams(self.tokens, self.n):
            self.ngram_counts[ngram] = self.ngram_counts.get(ngram, 0) + 1

    def predict(self, prompt: str) -> List[str]:
        vocab_size = len(self.unigram_counts)
        prompt_tokens = tokenize(prompt)[(-self.history - 1) :]
        prompt_tokens = [""] * max(0, self.n - 1 - len(prompt_tokens)) + prompt_tokens

        def mle(token: str) -> float:
            p_log = math.log(1.0)
            tokens = [*prompt_tokens, token]
            for ngram in ngrams(tokens, self.n):
                unigram_count = self.unigram_counts.get(ngram[0], 0)
                ngram_count = self.ngram_counts.get(ngram, 0)

                if self.smoothing is Smoothing.LAPLACE:
                    p_log = p_log + math.log((ngram_count + 1) / (unigram_count + vocab_size))
                elif self.smoothing is Smoothing.GOOD_TURING:
                    if unigram_count == 0:
                        p_log = p_log + math.log(1 / len(self.tokens))
                    else:
                        p_log = p_log + math.log((ngram_count or (1 / len(self.tokens))) / unigram_count)
                else:
                    if ngram_count == 0:
                        return 0
                    p_log = p_log + math.log(ngram_count / unigram_count)

            return math.exp(p_log)

        return sorted(self.unigram_counts.keys(), key=mle, reverse=True)


__all__ = ["MLE"]
