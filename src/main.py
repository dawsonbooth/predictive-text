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


def list_predictions(model: Model, prompt: str, num_predictions: int = 5) -> None:
    token_odds = model.predict(prompt)
    for i, item in enumerate(list(token_odds.items())):
        print(f"\t{i + 1}. '{item[0]}' ({item[1]})")
        if i + 1 == num_predictions:
            return


prompt = "The hitch hiker's guide to the"

if __name__ == "__main__":
    print(f"Prompt: {prompt}")

    print("NAIVE")
    m: Model = Naive(2, history=5)
    m.fit(text)
    list_predictions(m, prompt)
    free_write(m, prompt)

    print("KNN")
    m = KNN(2, {Distance.NAIVE, Distance.WU_PALMER})
    m.fit(text)
    list_predictions(m, prompt)
    free_write(m, prompt)
