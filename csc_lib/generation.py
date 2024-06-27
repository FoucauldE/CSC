import torch

class Generator:

    def __init__(self, tokenizer, model, device='cuda'):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def complete_prompt(self, prompt, gen_args, max_new_tokens):
        input = self.tokenizer(prompt, return_tensors='pt')
        input_ids, mask = input.input_ids, input.attention_mask

        if next(self.model.parameters()).is_cuda:
            input_ids, mask = input_ids.to('cuda'), mask.to('cuda')

        gen_args["input_ids"] = input_ids
        gen_args["attention_mask"] = mask
        gen_args["max_new_tokens"] = max_new_tokens

        output = self.model.generate(**gen_args)

        return self.tokenizer.decode(output.sequences[0])
