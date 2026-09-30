"""Entrada de texto pelo teclado com o mesmo contrato dos engines de STT."""

from engines.base_engine import BaseEngine


class KeyboardEngine(BaseEngine):
    """Permite informar manualmente o texto que seria produzido pelo STT."""

    def __init__(self, prompt: str = "Digite o texto (Enter para confirmar): "):
        self.prompt = prompt

    def transcribe(self) -> str:
        return input(self.prompt).strip()
