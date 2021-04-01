import collections
import enum
from collections import defaultdict
from functools import lru_cache
from typing import Collection, Dict, Tuple

import wn
import wn.similarity

from nlp import edit_distance, lemmatize, ngrams, pos_tags, simplify_tag, tokenize

from .model import Model


class Distance(enum.Enum):
    NAIVE = enum.auto()
    LENGTH = enum.auto()
    LEVENSHTEIN = enum.auto()
    PATH = enum.auto()
    WU_PALMER = enum.auto()
    LEACOCK_CHORDOROW = enum.auto()
    POS = enum.auto()


@lru_cache(maxsize=None)
def token_distance(token: str, other: str, metrics: Collection[Distance] = {Distance.NAIVE}) -> float:
    distance = 0.0
    token_lemma = lemmatize(token)
    other_lemma = lemmatize(other)

    if Distance.POS in metrics:
        token_pos = pos_tags([token])[0][1]
        other_pos = pos_tags([other])[0][1]
        distance += int(simplify_tag(token_pos) != simplify_tag(other_pos))
    if Distance.NAIVE in metrics:
        distance += int(token_lemma != other_lemma)
    if Distance.LENGTH in metrics:
        distance += abs(len(token_lemma) - len(other_lemma))
    if Distance.LEVENSHTEIN in metrics:
        distance += edit_distance(token_lemma, other_lemma)
    if any(d in metrics for d in {Distance.PATH, Distance.WU_PALMER, Distance.LEACOCK_CHORDOROW}):
        try:
            synset1, synset2 = wn.synsets(token_lemma)[0], wn.synsets(other_lemma)[0]
        except IndexError:
            distance += len([d in metrics for d in {Distance.PATH, Distance.WU_PALMER, Distance.LEACOCK_CHORDOROW}])
            return distance / len(metrics)
        if Distance.PATH in metrics:
            distance += 1 - wn.similarity.path(synset1, synset2)
        if Distance.WU_PALMER in metrics:
            distance += 1 - wn.similarity.wup(synset1, synset2)
        if Distance.LEACOCK_CHORDOROW in metrics:
            distance += 1 - wn.similarity.lch(synset1, synset2)

    return distance / len(metrics)


def ngram_distance(ngram: Tuple[str, ...], other: Tuple[str, ...], metrics: Collection[Distance] = {Distance.NAIVE}):
    return sum(token_distance(ngram[i], other[i], tuple(set(metrics))) for i in range(len(ngram))) / len(ngram)


class KNN(Model):
    __slots__ = "n", "similarity", "unigrams", "ngram_follower_counts"

    n: int
    metrics: Collection[Distance]

    ngram_follower_counts: Dict[Tuple[str, ...], Dict[str, int]]

    def __init__(self, n: int = 3, metrics: Collection[Distance] = {Distance.NAIVE}) -> None:
        super().__init__()
        self.n = n
        self.metrics = metrics

    def fit(self, text: str) -> None:
        tokens = tokenize(text)

        self.ngram_follower_counts = defaultdict(lambda: defaultdict(int))
        for ngram in ngrams(tokens, self.n + 1):
            ngram, follower = ngram[: self.n], ngram[-1]
            self.ngram_follower_counts[ngram][follower] += 1

    def predict(self, prompt: str) -> Dict[str, float]:
        prompt_ngram = tuple(tokenize(prompt))[-self.n :]
        prompt_ngram = ("",) * max(0, self.n - len(prompt_ngram)) + prompt_ngram

        follower_odds: Dict[str, int] = defaultdict(int)

        for neighbor in self.ngram_follower_counts.keys():
            neighbor_distance = ngram_distance(prompt_ngram, neighbor, self.metrics)
            for follower, follower_count in self.ngram_follower_counts[neighbor].items():
                follower_odds[follower] += follower_count * (self.n - neighbor_distance)

        return collections.OrderedDict((sorted(follower_odds.items(), key=lambda item: item[1], reverse=True)))


__all__ = ["KNN", "Distance", "ngram_distance"]
