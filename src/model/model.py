from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence


class Model(ABC):
    @abstractmethod
    def fit(self, text: Sequence[str]) -> None:
        ...

    @abstractmethod
    def predict(self, text: str) -> List[str]:
        ...
