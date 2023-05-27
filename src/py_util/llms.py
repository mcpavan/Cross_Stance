from typing import Any
from llama_cpp import Llama
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_llm_class_and_params(model_type, params):
    model_type = model_type.lower()
    if model_type == "llama_cpp":
        return Llama(model_path=params["model_path"])

class LlamaCpp_Model:
    def __init__(self, params, num_labels=2):
        self.model = Llama(model_path=params["model_path"])
        self.model_type = "llama_cpp"
        self.num_labels = num_labels
        self.output_dim = 1
    
    def __call__(self, *args: Any, **kwds: Any):
        return self.forward(*args, **kwds)
    
    def forward(self, prompt, params):
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            suffix: A suffix to append to the generated text. If None, no suffix is appended.
            max_tokens: The maximum number of tokens to generate.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for sampling.
            logprobs: The number of logprobs to return. If None, no logprobs are returned.
            echo: Whether to echo the prompt.
            stop: A list of strings to stop generation when encountered.
            repeat_penalty: The penalty to apply to repeated tokens.
            top_k: The top-k value to use for sampling.
            stream: Whether to stream the results.
        """

        output = self.model(prompt, **params)
        return output["choices"][0]["text"]

class HF_Llama_Model:
    def __init__(self, model="bigscience/bloom-1b7", hf_model_params={}, num_labels=2):
        # self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", **hf_model_params)
        self.model_type = "hugging_face"
        self.num_labels = num_labels
        self.output_dim = 1

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(args, kwds)
    
    def forward(self, prompt_ids, params):
        generated_ids = self.model.generate(
            prompt_ids,
            max_length=params.get("max_length", 32)
        )
        # self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return generated_ids