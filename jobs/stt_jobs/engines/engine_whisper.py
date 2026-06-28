"""Transcricao com Whisper via faster-whisper (grava e depois transcreve).

Diferente do Vosk, o Whisper nao e streaming nativo: a gente grava um
trecho inteiro e so entao manda transcrever.
"""
from faster_whisper import WhisperModel
import mic as mic
from engines.base_engine import BaseEngine

DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16"


class WhisperEngine(BaseEngine):

    def __init__(self, model_size: str = "small"):
        self.model_size = model_size

    def transcribe(self) -> str:
        print(f"Carregando Whisper '{self.model_size}' ({DEVICE}/{COMPUTE_TYPE})...")
        model = WhisperModel(self.model_size, device=DEVICE, compute_type=COMPUTE_TYPE)

        audio = mic.record_until_enter()
        print("Transcrevendo...\n")

        segments, info = model.transcribe(audio, language="pt")
        print(f"(idioma: {info.language}, confianca: {info.language_probability:.2f})\n")
        for seg in segments:
            print(seg.text.strip())
            return seg.text.strip()
