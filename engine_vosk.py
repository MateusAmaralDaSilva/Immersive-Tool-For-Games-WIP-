"""Transcricao em tempo real com Vosk (streaming).

O Vosk processa o audio em blocos enquanto voce fala:
- "Parcial" = palpite atual, ainda muda enquanto voce fala
- "Final"   = trecho que o Vosk ja considera fechado
"""
import json
from vosk import Model, KaldiRecognizer
import mic


def run(model_path):
    print(f"Carregando modelo Vosk de: {model_path}")
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, mic.SAMPLE_RATE)

    print("Pronto. Fale algo (Ctrl+C para sair).\n")
    try:
        for chunk in mic.stream_chunks():
            if recognizer.AcceptWaveform(chunk):
                texto = json.loads(recognizer.Result()).get("text", "")
                if texto:
                    print("Final  :", texto)
            else:
                parcial = json.loads(recognizer.PartialResult()).get("partial", "")
                print("Parcial:", parcial, end="          \r")
    except KeyboardInterrupt:
        print("\nEncerrado.")
