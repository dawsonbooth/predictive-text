from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict


class Model(ABC):
    @abstractmethod
    def fit(self, text: str) -> None:
        ...

    @abstractmethod
    def predict(self, text: str) -> Dict[str, float]:
        ...


__all__ = ["Model"]
