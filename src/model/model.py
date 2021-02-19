from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class Model(ABC):
    @abstractmethod
    def fit(self, text: str) -> None:
        ...

    @abstractmethod
    def predict(self, text: str) -> List[str]:
        ...
