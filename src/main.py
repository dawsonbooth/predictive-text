import itertools

from data import text
from model import MLE, Model
from model.mle import Smoothing


def free_write(model: Model, prompt: str, max_length: int = 80) -> None:
    while len(prompt) <= max_length:
        next_ = model.predict(prompt)[0]
        print(next_, end=" ")
        prompt += f" {next_}"
    print()


n_sizes = [2, 3, 4]
smoothings = [Smoothing.NONE, Smoothing.LAPLACE, Smoothing.GOOD_TURING]
histories = [10]
prompts = ["Hitch hiker's guide to the", "It begins with a"]


if __name__ == "__main__":
    for n, smoothing, history, prompt in itertools.product(n_sizes, smoothings, histories, prompts):
        print((n, smoothing.name, history, prompt))
        m = MLE(n, smoothing, history)
        m.fit(text)
        free_write(m, prompt)
