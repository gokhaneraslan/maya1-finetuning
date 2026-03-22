import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import TrainConfig
from .constants import PAD_TOKEN_ID


def setup_model(config: TrainConfig):

    print(f"Loading model from '{config.model_path}'...")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
        #attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True,
    )

    tokenizer.pad_token_id = PAD_TOKEN_ID

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    
    for param in model.parameters():
        param.requires_grad = True

    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{total:.0f}M parameters - all trainable.")

    return model, tokenizer