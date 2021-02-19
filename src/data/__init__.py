from pathlib import Path

from nlp import tokenize

text = (Path(__file__).parent / "homer-illiad.txt").read_text()
tokens = tokenize(text)
