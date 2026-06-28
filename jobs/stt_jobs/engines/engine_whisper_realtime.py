"""Whisper em tempo real (quase) usando deteccao de fala por energia (VAD).

Whisper nao e streaming nativo. O truque aqui:
  1. Escuta o microfone continuamente, em blocos de 0,1 s.
  2. Detecta quando voce esta falando, medindo a energia (volume) do audio.
  3. Quando voce faz uma pausa (silencio), transcreve so aquele trecho.

Resultado: o texto aparece logo apos cada frase, sem esperar voce terminar
tudo. Nao e palavra-por-palavra como o Vosk, mas fica bem responsivo.
"""
import numpy as np
from faster_whisper import WhisperModel
import mic as mic
from engines.base_engine import BaseEngine

DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16"

SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 0.8
MIN_SPEECH_DURATION = 0.3


def _rms(audio):
    return float(np.sqrt(np.mean(audio ** 2)))


class WhisperRealtimeEngine(BaseEngine):

    def __init__(self, model_size: str = "small"):
        self.model_size = model_size

    def transcribe(self) -> str:
        print(f"Carregando Whisper '{self.model_size}' ({DEVICE}/{COMPUTE_TYPE})...")
        model = WhisperModel(self.model_size, device=DEVICE, compute_type=COMPUTE_TYPE)

        blocks_per_second = mic.SAMPLE_RATE / mic.STREAM_BLOCK
        silence_limit = int(SILENCE_DURATION * blocks_per_second)
        min_speech_samples = int(MIN_SPEECH_DURATION * mic.SAMPLE_RATE)

        def transcrever(audio):
            segments, _ = model.transcribe(audio, language="pt")
            texto = " ".join(s.text.strip() for s in segments).strip()
            if texto:
                print(">", texto)

        print("Pronto. Fale; a transcricao aparece a cada pausa (Ctrl+C para sair).\n")

        buffer = []
        silence_blocks = 0
        speaking = False

        try:
            for bloco in mic.stream_float_chunks():
                if _rms(bloco) >= SILENCE_THRESHOLD:
                    buffer.append(bloco)
                    silence_blocks = 0
                    speaking = True
                elif speaking:
                    buffer.append(bloco)
                    silence_blocks += 1
                    if silence_blocks >= silence_limit:
                        audio = np.concatenate(buffer)
                        if len(audio) >= min_speech_samples:
                            transcrever(audio)
                        buffer = []
                        silence_blocks = 0
                        speaking = False
        except KeyboardInterrupt:
            full_text = " ".join(s.text.strip() for s in model.transcribe(np.concatenate(buffer), language="pt")[0]).strip()
            print("\nEncerrado.")
            return full_text.strip()
