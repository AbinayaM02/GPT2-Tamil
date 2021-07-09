# -*- coding: utf-8 -*-
import locale
print(locale.getpreferredencoding())


from transformers import AutoConfig, AutoModelForCausalLM,pipeline,AutoTokenizer
from datasets import load_dataset

MODEL_DIR = "/home/deepak/sources/gpt2-tamil/gpt2-tamil/"





#get prompt from dataset, will be replaced by manual prompt once I figure out how to render tamil font
dataset = load_dataset("oscar", "unshuffled_deduplicated_ta", split="train")
id =232
print(dataset[id]['text'])
tamil_prompt =dataset[id]['text']

# Get configuration and the model
config = AutoConfig.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_config(config)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)


generator= pipeline('text-generation', model=model, tokenizer=tokenizer)
model_output = generator(tamil_prompt, max_length=30, num_return_sequences=5)
print(model_output)

