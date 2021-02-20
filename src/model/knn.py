import enum
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import wn
import wn.similarity
from tqdm import tqdm

from nlp import lemmatize, ngrams, tokenize

from .model import Model


class Similarity(enum.Enum):
    NONE = enum.auto()
    PATH = enum.auto()
    WU_PALMER = enum.auto()
    LEACOCK_CHORDOROW = enum.auto()
    LEVENSHTEIN = enum.auto()


def ngram_similarity(ngram: Tuple[str, ...], other: Tuple[str, ...], similarity: Similarity = None):
    similarity = similarity or Similarity.NONE
    n = len(ngram)
    score = 0.0
    for i in range(n):
        lemma1, lemma2 = lemmatize(ngram[i]), lemmatize(other[i])
        if similarity is Similarity.NONE:
            score += int(lemma1 == lemma2)
        else:
            try:
                synset1, synset2 = wn.synsets(lemma1)[0], wn.synsets(lemma2)[0]
            except IndexError:
                continue
            if similarity is Similarity.PATH:
                score += wn.similarity.path(synset1, synset2)
            elif similarity is Similarity.WU_PALMER:
                score += wn.similarity.wup(synset1, synset2)
            elif similarity is Similarity.LEACOCK_CHORDOROW:
                score += wn.similarity.lch(synset1, synset2)
    return score / n


class KNN(Model):
    __slots__ = "n", "similarity", "unigrams", "ngram_follower_counts"

    n: int
    similarity: Similarity

    unigrams: Set[str]
    ngram_follower_counts: Dict[Tuple[str, ...], Dict[str, int]]

    def __init__(self, n: int = 2, similarity: Optional[Similarity] = None) -> None:
        super().__init__()
        self.n = n
        self.similarity = similarity or Similarity.NONE

    def fit(self, text: str) -> None:
        tokens = tokenize(text)

        self.unigrams = set(tokens)

        self.ngram_follower_counts = defaultdict(lambda: defaultdict(int))
        for ngram in ngrams(tokens, self.n + 1):
            ngram, follower = ngram[: self.n], ngram[-1]
            self.ngram_follower_counts[ngram][follower] += 1

    def predict(self, prompt: str) -> List[str]:
        prompt_ngram = tuple(tokenize(prompt))[-self.n :]
        prompt_ngram = ("",) * max(0, self.n - len(prompt_ngram)) + prompt_ngram

        follower_scores: Dict[str, int] = defaultdict(int)

        for neighbor in tqdm(self.ngram_follower_counts.keys()):
            neighbor_similarity = ngram_similarity(prompt_ngram, neighbor, self.similarity)
            for follower, follower_count in self.ngram_follower_counts[neighbor].items():
                follower_scores[follower] += follower_count * neighbor_similarity

        return sorted(self.unigrams, key=lambda token: follower_scores[token], reverse=True)


__all__ = ["KNN"]
