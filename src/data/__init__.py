from pathlib import Path

import pandas as pd

tweets = pd.read_csv(Path(__file__).parent / "Donald-Tweets!.csv")["Tweet_Text"]
