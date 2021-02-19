from typing import List

from data import tokens
from model import LaplaceBigrams, Model, NaiveBigrams

prompts = ["Sing to me", "of the", "Achilles", "to ever"]
models: List[Model] = [NaiveBigrams(), LaplaceBigrams()]

if __name__ == "__main__":
    for m in models:
        print(m.__class__.__name__)
        m.fit(tokens)
        for prompt in prompts:
            print(f"{prompt} : {m.predict(prompt)[:5]}")
