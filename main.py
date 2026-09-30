import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Os submodulos de STT e geracao importam pela pasta (ex.: `import mic`,
# `from engines...`, `from models...`), entao precisamos deixar essas pastas
# visiveis no sys.path antes de importa-los.
sys.path.append(str(BASE_DIR / "jobs" / "stt_jobs"))
sys.path.append(str(BASE_DIR / "jobs" / "gen_jobs"))

from jobs.stt_jobs.stt import speech_to_text
from jobs.stt_jobs.engines.engine_vosk import VoskEngine
from jobs.stt_jobs.engines.engine_whisper import WhisperEngine
from jobs.stt_jobs.engines.engine_whisper_realtime import WhisperRealtimeEngine
from jobs.stt_jobs.engines.engine_keyboard import KeyboardEngine

from jobs.gen_jobs.gen import generate_text
from jobs.gen_jobs.models.qwen_model import QwenModel
from jobs.gen_jobs.models.mistral_model import MistralModel
from jobs.gen_jobs.models.phi_model import PhiModel
from jobs.gen_jobs.models.deepseek_model import DeepSeekModel
from jobs.gen_jobs.prompt.prompt import PromptData, build_prompt

# Caminhos dos modelos
VOSK_MODEL_PATH = str(BASE_DIR / "models" / "stt_models" / "vosk" / "vosk-model-small-pt-0.3")
QWEN_MODEL_PATH = str(BASE_DIR / "models" / "gen_models" / "Qwen" / "Qwen3-4B-Q4_K_M.gguf")


def choose_stt_engine():
    print("Escolha a entrada de texto:")
    print("1 - Teclado")
    print("2 - Vosk")
    print("3 - Whisper")
    print("4 - Whisper em tempo real")

    option = input("Opcao [1]: ").strip() or "1"
    if option == "1":
        return KeyboardEngine()
    if option == "2":
        return VoskEngine(VOSK_MODEL_PATH)
    if option == "3":
        return WhisperEngine(model_size="small")
    if option == "4":
        return WhisperRealtimeEngine(model_size="small")

    raise ValueError("Opcao de entrada invalida. Escolha um numero de 1 a 4.")


if __name__ == "__main__":
    engine = choose_stt_engine()
    text = speech_to_text(engine)
    print(f"Texto reconhecido: {text}")

    # Escolha o modelo de geracao instanciando a classe desejada:
    #   QwenModel(...) | MistralModel(...) | PhiModel(...) | DeepSeekModel(...)
    model = QwenModel(model_path=QWEN_MODEL_PATH)
    prompt = build_prompt(PromptData(
        npc_id="gareth",
        player_id="ladino_01",
        player_input=text,
    ))
    state = generate_text(prompt, model)
    print(f"Estado escolhido: {state}")
