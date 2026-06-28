from engines.base_engine import BaseEngine


def speech_to_text(engine: BaseEngine) -> str:
    return engine.transcribe()

