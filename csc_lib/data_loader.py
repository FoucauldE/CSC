import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_models(model_path, base_model_name='bigscience/bloom-1b1'):

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

    # Load fine-tuned model
    ft_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    ft_model = PeftModel.from_pretrained(ft_model, model_path)

    # Move models to CUDA
    base_model.to('cuda')
    ft_model.to('cuda')

    return tokenizer, base_model, ft_model