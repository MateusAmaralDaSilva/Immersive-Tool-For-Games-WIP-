"""Ponto de entrada: escolhe qual motor de transcricao rodar.

Uso:
    python main.py
"""

# ----------------- Configuracao (mude aqui) -----------------
# Baixe o modelo Vosk em https://alphacephei.com/vosk/models
# e aponte o caminho para a pasta descompactada:
VOSK_MODEL_PATH = "models/vosk-model-small-pt-0.3"

# Tamanhos: tiny, base, small, medium, large-v3, large-v3-turbo
# O faster-whisper baixa sozinho na primeira vez.
WHISPER_MODEL_SIZE = "small"
# ------------------------------------------------------------


def main():
    print("Qual motor voce quer testar?")
    print("  1 - Vosk             (tempo real / streaming)")
    print("  2 - Whisper          (grava e transcreve)")
    print("  3 - Whisper tempo real (transcreve a cada pausa)")
    escolha = input("Opcao: ").strip()

    if escolha == "1":
        import engine_vosk
        engine_vosk.run(VOSK_MODEL_PATH)
    elif escolha == "2":
        import engine_whisper
        engine_whisper.run(WHISPER_MODEL_SIZE)
    elif escolha == "3":
        import engine_whisper_realtime
        engine_whisper_realtime.run(WHISPER_MODEL_SIZE)
    else:
        print("Opcao invalida.")


if __name__ == "__main__":
    main()
