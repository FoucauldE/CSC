import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_models(model_path, base_model_name='bigscience/bloom-1b1'):
    """Load models with or without PEFT depending on the presence of adapter_config.json.
    Move models to GPU if available"""

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

    # Load fine-tuned model
    if os.path.isfile(os.path.join(model_path, 'adapter_config.json')):
        # Using PEFT if specified path contains adapter_config.json file
        ft_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        ft_model = PeftModel.from_pretrained(ft_model, model_path)
    else:
        # Load fully fine-tuned model directly else
        ft_model = AutoModelForCausalLM.from_pretrained(model_path)

    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_model.to(device)
    ft_model.to(device)

    return tokenizer, base_model, ft_model