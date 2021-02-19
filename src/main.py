import itertools

from data import text
from model import Model, NGrams
from model.ngrams import Smoothing


def free_write(model: Model, prompt: str, max_length: int = 80) -> None:
    while len(prompt) <= max_length:
        next_ = model.predict(prompt)[0]
        print(next_, end=" ")
        prompt += f" {next_}"
    print()


n_sizes = [2, 3, 4]
smoothings = [Smoothing.NONE, Smoothing.LAPLACE, Smoothing.GOOD_TURING]
prompts = ["Go, and I will tell you", "On Sunday, the other gods"]


if __name__ == "__main__":
    for n, smoothing, prompt in itertools.product(n_sizes, smoothings, prompts):
        print((n, smoothing.name, prompt))
        m = NGrams(n, smoothing)
        m.fit(text)
        free_write(m, prompt)
