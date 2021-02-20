import itertools

from data import text
from model import KNN, MLE, Model
from model.knn import Similarity
from model.mle import Smoothing


def free_write(model: Model, prompt: str, max_length: int = 80) -> None:
    while len(prompt) <= max_length:
        next_ = model.predict(prompt)[0]
        print(next_, end=" ")
        prompt += f" {next_}"
    print()


def mle():
    n_sizes = [2, 3]
    smoothings = [Smoothing.NONE, Smoothing.LAPLACE, Smoothing.GOOD_TURING]
    histories = [10]
    prompts = ["Hitch hiker's guide to the", "It begins with a"]

    for n, smoothing, history, prompt in itertools.product(n_sizes, smoothings, histories, prompts):
        print((n, smoothing.name, history, prompt))
        m = MLE(n, smoothing, history)
        m.fit(text)
        free_write(m, prompt)


def knn():
    n_sizes = [3, 4]
    similarities = [Similarity.NONE, Similarity.PATH, Similarity.WU_PALMER, Similarity.LEACOCK_CHORDOROW]
    prompts = ["Hitch hiker's guide to the", "It begins with a"]

    for n, similarity, prompt in itertools.product(n_sizes, similarities, prompts):
        print((n, similarity.name, prompt))
        m = KNN(n, similarity)
        m.fit(text)
        free_write(m, prompt)


if __name__ == "__main__":
    # mle()
    knn()
