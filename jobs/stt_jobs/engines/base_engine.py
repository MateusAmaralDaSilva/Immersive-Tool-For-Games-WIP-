from abc import ABC, abstractmethod


class BaseEngine(ABC):

    @abstractmethod
    def transcribe(self) -> str:
        pass
