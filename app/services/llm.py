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

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        """
        Generate response from the LLM given a prompt.
        Returns only the generated answer (without prompt echo).
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=True
            )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated answer
        if "Answer:" in full_response:
            answer = full_response.split("Answer:")[-1].strip()
        else:
            answer = full_response.strip()

        return answer
