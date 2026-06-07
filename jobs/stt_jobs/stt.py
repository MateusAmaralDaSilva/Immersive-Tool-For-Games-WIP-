from jobs.stt_jobs.engines import engine_whisper
from jobs.stt_jobs.engines import engine_vosk, engine_whisper_realtime


VOSK_MODEL_PATH = "models/stt_models/vosk/vosk-model-small-pt-0.3"

# Tamanhos: tiny, base, small, medium, large-v3, large-v3-turbo
# O faster-whisper baixa sozinho na primeira vez.
WHISPER_MODEL_SIZE = "small"
# ------------------------------------------------------------


def stt(escolha=None):

    if escolha == "1":
        texto =engine_vosk.run(VOSK_MODEL_PATH)
    elif escolha == "2":
        texto = engine_whisper.run(WHISPER_MODEL_SIZE)
    elif escolha == "3":
        texto = engine_whisper_realtime.run(WHISPER_MODEL_SIZE)
    elif escolha == "4":
        texto = input("Insira o texto: ")
    else:
        print("Opcao invalida.")
    return texto.strip()

if __name__ == "__main__":
    stt(escolha="3")
