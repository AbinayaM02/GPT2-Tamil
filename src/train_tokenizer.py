from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer  # Tokenizer, normalizers, trainers

model_dir = "./gpt2-tamil"  # ${MODEL_DIR}

# load dataset
dataset = load_dataset("oscar", "unshuffled_deduplicated_ta", split="train")

# Instantiate tokenizer
tokenizer = ByteLevelBPETokenizer()


def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]


# Customized training
tokenizer.train_from_iterator(
    batch_iterator(),
    vocab_size=50265,
    min_frequency=2,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ],
)

# Save files to disk
tokenizer.save(f"{model_dir}/tokenizer.json")
