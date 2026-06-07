from llama_cpp import Llama


class QwenModel:

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1
    ):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )

    def predict(self, prompt: str) -> str:

        output = self.llm(
            prompt,
            max_tokens=10,
            temperature=0.1,
            stop=["\n"]
        )

        return output["choices"][0]["text"].strip()