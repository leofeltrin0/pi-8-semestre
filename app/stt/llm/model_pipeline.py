import transformers
import torch

class TextGenerationPipeline:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def summarize(self, msg: str, max_tokens: int = 256) -> str:
        prompt = [
            {"role": "system", "content": "Leia atentamente a transcrição a seguir e resuma brevemente o contexto da mensagem:"},
            {"role": "user", "content": msg},
        ]
        outputs = self.pipeline(prompt, max_new_tokens=max_tokens)
        return outputs[0]["generated_text"]
    
    def interact_with_llm(self, msg: str, max_tokens: int = 256) -> str:
        prompt = [
        {"role": "system", "content": "Você é um assistente avançado que pode responder perguntas, realizar análises e oferecer explicações detalhadas."},
        {"role": "user", "content": msg},
        ]
        outputs = self.pipeline(prompt, max_new_tokens=max_tokens)
        return outputs[0]["generated_text"]
