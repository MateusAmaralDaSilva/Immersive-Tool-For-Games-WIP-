"""Whisper em tempo real (quase) usando deteccao de fala por energia (VAD).

Whisper nao e streaming nativo. O truque aqui:
  1. Escuta o microfone continuamente, em blocos de 0,1 s.
  2. Detecta quando voce esta falando, medindo a energia (volume) do audio.
  3. Quando voce faz uma pausa (silencio), transcreve so aquele trecho.

Resultado: o texto aparece logo apos cada frase, sem esperar voce terminar
tudo. Nao e palavra-por-palavra como o Vosk, mas fica bem responsivo.

Obs: a transcricao roda no mesmo loop da captura. Para um teste tudo bem,
mas para producao voce passaria a transcricao para uma thread separada.
"""
import numpy as np
from faster_whisper import WhisperModel
import jobs.stt_jobs.mic as mic

# Mesmo hardware do engine_whisper.py:
DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16"

# Ajuste fino da deteccao de fala:
SILENCE_THRESHOLD = 0.01   # energia abaixo disso = silencio (suba se houver ruido)
SILENCE_DURATION = 0.8     # segundos de pausa para encerrar a frase
MIN_SPEECH_DURATION = 0.3  # ignora trechos curtos demais (cliques, ruidos)


def _rms(audio):
    """Energia (volume) do bloco: raiz da media dos quadrados."""
    return float(np.sqrt(np.mean(audio ** 2)))


def run(model_size="small"):
    print(f"Carregando Whisper '{model_size}' ({DEVICE}/{COMPUTE_TYPE})...")
    model = WhisperModel(model_size, device=DEVICE, compute_type=COMPUTE_TYPE)

    # Converte as duracoes (em segundos) para quantidade de blocos/amostras.
    blocks_per_second = mic.SAMPLE_RATE / mic.STREAM_BLOCK
    silence_limit = int(SILENCE_DURATION * blocks_per_second)
    min_speech_samples = int(MIN_SPEECH_DURATION * mic.SAMPLE_RATE)

    def transcrever(audio):
        segments, _ = model.transcribe(audio, language="pt")
        texto = " ".join(s.text.strip() for s in segments).strip()
        if texto:
            print(">", texto)

    print("Pronto. Fale; a transcricao aparece a cada pausa (Ctrl+C para sair).\n")

    buffer = []          # blocos de fala acumulados
    silence_blocks = 0   # blocos seguidos de silencio
    speaking = False

    try:
        for bloco in mic.stream_float_chunks():
            if _rms(bloco) >= SILENCE_THRESHOLD:
                # Esta falando: acumula e zera o contador de silencio.
                buffer.append(bloco)
                silence_blocks = 0
                speaking = True
            elif speaking:
                # Pausa depois de ter falado: conta o silencio.
                buffer.append(bloco)  # guarda o rabicho de silencio tambem
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
