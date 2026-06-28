from models.base_model import BaseModel


class MistralModel(BaseModel):

    def __init__(self, model_path: str, n_ctx: int = 8192, n_gpu_layers: int = -1):
        super().__init__(model_path, n_ctx, n_gpu_layers)

    def format_prompt(self, prompt: str) -> str:
        return f"[INST] {prompt} [/INST]"
