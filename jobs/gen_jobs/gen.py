from models.base_model import BaseModel


def generate_text(prompt: str, model: BaseModel) -> str:
    return model.generate(prompt)
