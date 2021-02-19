import math
from typing import Dict, List, Sequence, Tuple

from nlp import ngrams, tokenize

from ..model import Model


class LaplaceBigrams(Model):
    unigrams: Dict[str, int]
    bigrams: Dict[Tuple[str, ...], int]

    def fit(self, tokens: Sequence[str]) -> None:
        self.unigrams = dict()
        for unigram in tokens:
            self.unigrams[unigram] = self.unigrams.get(unigram, 0) + 1

        self.bigrams = dict()
        for bigram in ngrams(tokens, 2):
            self.bigrams[bigram] = self.bigrams.get(bigram, 0) + 1

    def predict(self, prompt: str) -> List[str]:
        vocab_size = len(self.unigrams)
        prompt_tokens = tokenize(prompt)

        def prob_follows_prompt(token: str) -> float:
            p_log = math.log(1.0)
            for bigram in ngrams([*prompt_tokens, token], 2):
                unigram_count = self.unigrams.get(bigram[0], 0)
                bigram_count = self.bigrams.get(bigram, 0)

                p_log = p_log + math.log((bigram_count + 1) / (unigram_count + vocab_size))

            return math.exp(p_log)

        return sorted(self.unigrams.keys(), key=prob_follows_prompt, reverse=True)
