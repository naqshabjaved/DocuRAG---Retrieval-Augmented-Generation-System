import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMService:
    """
    Handles loading and inference of the LLM.
    """

    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name

        print("Loading LLM... This may take a moment.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            device_map="cpu"
        )

        self.model.eval()

        print("LLM loaded successfully.")

    def generate(self, prompt: str, max_new_tokens: int = 250) -> str:
        messages = [
            {"role": "system", "content": "Answer strictly using the provided context."},
            {"role": "user", "content": prompt}
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=False
            )

        generated_tokens = outputs[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()

