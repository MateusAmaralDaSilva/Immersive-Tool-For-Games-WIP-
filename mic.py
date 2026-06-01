"""Captura de audio do microfone.

Vosk e Whisper precisam de coisas diferentes, entao temos duas funcoes:
- stream_chunks()       -> blocos continuos de audio cru (para Vosk, streaming)
- record_until_enter()  -> grava tudo ate apertar Enter (para Whisper, batch)
"""
import sounddevice as sd
import numpy as np

# 16 kHz e o padrao esperado tanto pelo Vosk quanto pelo Whisper.
SAMPLE_RATE = 16000

# Tamanho do bloco lido por vez no modo tempo real (em amostras).
# 1600 amostras @ 16 kHz = 0,1 s por bloco -> deteccao de pausa mais granular.
STREAM_BLOCK = 1600


def stream_chunks(blocksize=4000):
    """Abre o microfone e gera blocos de audio cru (int16) continuamente."""
    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=blocksize * 2,
        dtype="int16",
        channels=1,
    )
    stream.start()
    try:
        while True:
            data, _ = stream.read(blocksize)
            yield bytes(data)
    finally:
        stream.stop()
        stream.close()


def record_until_enter():
    """Grava do microfone ate o usuario apertar Enter.

    Devolve um array numpy float32 mono (formato que o Whisper aceita direto).
    """
    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", callback=callback):
        print("Gravando... pressione Enter para parar.")
        input()

    return np.concatenate(frames, axis=0).flatten()


def stream_float_chunks():
    """Gera blocos de audio float32 mono continuamente.

    Usado pelo Whisper em tempo real, que precisa de float32 (e nao int16
    como o Vosk) para medir energia e transcrever.
    """
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=STREAM_BLOCK,
        channels=1,
        dtype="float32",
    )
    stream.start()
    try:
        while True:
            data, _ = stream.read(STREAM_BLOCK)
            yield data.flatten()
    finally:
        stream.stop()
        stream.close()
