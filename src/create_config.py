from transformers import GPT2Config

model_dir = "./gpt2-tamil"  # ${MODEL_DIR}

config = GPT2Config.from_pretrained(
    "gpt2", resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0
)
config.save_pretrained(model_dir)
