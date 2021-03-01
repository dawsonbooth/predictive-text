import collections
import enum
from collections import defaultdict
from typing import Dict, Optional, Tuple

import wn
import wn.similarity

from nlp import edit_distance, lemmatize, ngrams, pos_tags, tokenize

from .model import Model


class Distance(enum.Enum):
    NONE = enum.auto()
    LENGTH = enum.auto()
    LEVENSHTEIN = enum.auto()
    PATH = enum.auto()
    WU_PALMER = enum.auto()
    LEACOCK_CHORDOROW = enum.auto()
    POS = enum.auto()


def ngram_distance(ngram: Tuple[str, ...], other: Tuple[str, ...], metric: Distance = Distance.NONE):
    distance = 0.0
    ngram_lemmas = tuple(lemmatize(t) for t in ngram)
    other_lemmas = tuple(lemmatize(t) for t in other)

    if metric is Distance.POS:
        ngram_pos = tuple(tag[1] for tag in pos_tags(ngram_lemmas))
        other_pos = tuple(tag[1] for tag in pos_tags(other_lemmas))
        for i in range(len(ngram)):
            distance += int(ngram_pos[i] != other_pos[i])
        return distance

    for i in range(len(ngram)):
        if metric is Distance.NONE:
            distance += int(ngram_lemmas[i] != other_lemmas[i])
        elif metric is Distance.LENGTH:
            distance += abs(len(ngram_lemmas[i]) - len(other_lemmas[i]))
        elif metric is Distance.LEVENSHTEIN:
            distance += edit_distance(ngram_lemmas[i], other_lemmas[i])
        else:
            try:
                synset1, synset2 = wn.synsets(ngram_lemmas[i])[0], wn.synsets(other_lemmas[i])[0]
            except IndexError:
                continue
            if metric is Distance.PATH:
                distance += 1 - wn.similarity.path(synset1, synset2)
            elif metric is Distance.WU_PALMER:
                distance += 1 - wn.similarity.wup(synset1, synset2)
            elif metric is Distance.LEACOCK_CHORDOROW:
                distance += 1 - wn.similarity.lch(synset1, synset2)
    return distance


class KNN(Model):
    __slots__ = "n", "similarity", "unigrams", "ngram_follower_counts"

    n: int
    metric: Distance

    ngram_follower_counts: Dict[Tuple[str, ...], Dict[str, int]]

    def __init__(self, n: int = 3, metric: Optional[Distance] = None) -> None:
        super().__init__()
        self.n = n
        self.metric = metric or Distance.NONE

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
            neighbor_distance = ngram_distance(prompt_ngram, neighbor, self.metric)
            for follower, follower_count in self.ngram_follower_counts[neighbor].items():
                follower_odds[follower] += (1 - (1 / follower_count)) * (self.n - neighbor_distance)

        return collections.OrderedDict((sorted(follower_odds.items(), key=lambda item: item[1], reverse=True)))


__all__ = ["KNN"]
