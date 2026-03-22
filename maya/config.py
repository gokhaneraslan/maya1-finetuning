import yaml
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class TrainConfig:

    model_path: str = "maya-research/maya1"
    dataset_dir: str = "./data/wavs"
    metadata_path: str = "./data/metadata_final.json"
    preprocessed_dir: str = "./data/preprocessed"
    output_dir: str = "./output/maya1_finetune"

    resume_from_checkpoint: Optional[str] = None

    max_text_len: int = 400
    max_speech_len: int = 3000

    batch_size: int = 4
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-5
    num_epochs: int = 100
    warmup_steps: int = 560
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    fp16: bool = False
    bf16: bool = True

    logging_steps: int = 10
    save_steps: int = 560
    save_total_limit: int = 15

    default_description: str = "Turkish narration, neutral tone, clear voice."

    sample_text: str = "Merhaba, bugün hava gerçekten çok güzel değil mi? Umarım her şey yolundadır."
    sample_description: str = (
        "Realistic female voice in her 30s with a Turkish accent. "
        "Deep pitch, warm timbre, conversational pacing, neutral tone, narrator role."
    )


def load_config(path: str) -> TrainConfig:
    
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    valid_fields = TrainConfig.__dataclass_fields__.keys()
    filtered = {k: v for k, v in data.items() if k in valid_fields}

    unknown = set(data) - set(valid_fields)
    if unknown:
        print(f"Warning: unknown keys ignored -> {unknown}")

    return TrainConfig(**filtered)


def save_config(cfg: TrainConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(asdict(cfg), f, allow_unicode=True, sort_keys=False)