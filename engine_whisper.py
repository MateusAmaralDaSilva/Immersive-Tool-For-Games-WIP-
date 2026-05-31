"""Transcricao com Whisper via faster-whisper (grava e depois transcreve).

Diferente do Vosk, o Whisper nao e streaming nativo: a gente grava um
trecho inteiro e so entao manda transcrever.
"""
from faster_whisper import WhisperModel
import mic

# Ajuste conforme seu hardware:
#   device="cuda" + compute_type="int8_float16" -> ideal para a RTX 2060 (~4 GB VRAM)
#   device="cpu"  + compute_type="int8"          -> se nao tiver GPU NVIDIA
DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16"


def run(model_size="small"):
    print(f"Carregando Whisper '{model_size}' ({DEVICE}/{COMPUTE_TYPE})...")
    model = WhisperModel(model_size, device=DEVICE, compute_type=COMPUTE_TYPE)

    audio = mic.record_until_enter()
    print("Transcrevendo...\n")

    segments, info = model.transcribe(audio, language="pt")
    print(f"(idioma: {info.language}, confianca: {info.language_probability:.2f})\n")
    for seg in segments:
        print(seg.text.strip())
