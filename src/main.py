import random

from data import text
from model import KNN, Model, Naive
from model.knn import Distance


def free_write(model: Model, prompt: str, max_length: int = 80, temperature: int = 2) -> None:
    while len(prompt) <= max_length:
        token_odds = model.predict(prompt)
        tokens, odds = list(token_odds.keys()), list(token_odds.values())
        next_ = random.choices(tokens[:temperature], weights=odds[:temperature], k=1)[0]
        print(next_, end=" ")
        prompt += f" {next_}"
    print()


prompt = "The hitch hiker's guide"

if __name__ == "__main__":
    print(f"Naive: {prompt}")
    m: Model = Naive(2, history=5)
    m.fit(text)
    print(list(m.predict(prompt).items())[:5])
    free_write(m, prompt)

    print(f"KNN: {prompt}")
    m = KNN(3, {Distance.NAIVE, Distance.POS})
    m.fit(text)
    print(list(m.predict(prompt).items())[:5])
    free_write(m, prompt)
