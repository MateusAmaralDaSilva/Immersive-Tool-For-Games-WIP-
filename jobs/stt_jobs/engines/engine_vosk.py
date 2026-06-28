"""Transcricao em tempo real com Vosk (streaming).

O Vosk processa o audio em blocos enquanto voce fala:
- "Parcial" = palpite atual, ainda muda enquanto voce fala
- "Final"   = trecho que o Vosk ja considera fechado
"""
import json
from vosk import Model, KaldiRecognizer
import mic as mic
from engines.base_engine import BaseEngine


class VoskEngine(BaseEngine):

    def __init__(self, model_path: str):
        self.model_path = model_path

    def transcribe(self) -> str:
        print(f"Carregando modelo Vosk de: {self.model_path}")
        model = Model(self.model_path)
        recognizer = KaldiRecognizer(model, mic.SAMPLE_RATE)
        full_text = ""
        print("Pronto. Fale algo (Ctrl+C para sair).\n")
        try:
            for chunk in mic.stream_chunks():
                if recognizer.AcceptWaveform(chunk):
                    texto = json.loads(recognizer.Result()).get("text", "")
                    if texto:
                        full_text += texto + " "
                        print("Final  :", texto)
                else:
                    parcial = json.loads(recognizer.PartialResult()).get("partial", "")
                    print("Parcial:", parcial, end="\r")

        except KeyboardInterrupt:
            print("\nEncerrado.")
            return full_text.strip()
