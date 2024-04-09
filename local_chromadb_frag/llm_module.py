# llm_module.py
import os
import sys
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

class LLMModule:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipelines = {}


    def load_model(self):
        model_config = transformers.AutoConfig.from_pretrained(self.model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Set up quantization config
        use_4bit = True
        bnb_4bit_compute_dtype = "float16"
        bnb_4bit_quant_type = "nf4"
        use_nested_quant = False

        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_nested_quant,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
        )

    def load_pipelines(self):
        # Initialize the standalone query generation pipeline
        self.pipelines["standalone_question"] = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=0.0,
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=1000,
        )

        # Initialize the response generation pipeline
        self.pipelines["response"] = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=0.2,
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=1000,
        )

    def generate_text(self, prompt, task="response"):
        try:
            pipeline = self.pipelines.get(task)
            if pipeline:
                response = pipeline(prompt)[0]["generated_text"]
                return response
            else:
                raise ValueError(f"Invalid task: {task}")
        except Exception as e:
            print("Error:")
            print(str(e))
            raise e



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please provide a query as a command line argument.")
        sys.exit(1)

    query = sys.argv[1]

    llm = LLMModule(model_name='mistralai/Mistral-7B-Instruct-v0.1')
    llm.load_model()
    llm.load_pipelines()

    response = llm.generate_text(query)
    print(response)
