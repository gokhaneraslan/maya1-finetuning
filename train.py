import os
import argparse

import torch
from snac import SNAC
from transformers import Trainer, TrainingArguments

from maya.config import TrainConfig, load_config, save_config
from maya.model import setup_model
from maya.dataset import Maya1Dataset, data_collator
from maya.callback import AudioSampleCallback
from maya.constants import SNAC_MODEL_NAME


def train(config: TrainConfig):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, tokenizer = setup_model(config)

    print("Loading SNAC helper for sample generation")
    snac_model = SNAC.from_pretrained(SNAC_MODEL_NAME).eval().to(device)

    train_dataset = Maya1Dataset(config)

    callback = AudioSampleCallback(
        output_dir=config.output_dir,
        snac_model=snac_model,
        tokenizer=tokenizer,
        sample_text=config.sample_text,
        sample_description=config.sample_description,
        device=device,
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,

        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=True,

        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        lr_scheduler_type="cosine",

        fp16=config.fp16,
        bf16=config.bf16,
        tf32=True,

        logging_strategy="steps",
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,

        eval_strategy="no",
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_num_workers=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[callback],
    )

    os.makedirs(config.output_dir, exist_ok=True)
    save_config(config, os.path.join(config.output_dir, "active_config.yaml"))

    resume = config.resume_from_checkpoint
    if resume and os.path.isdir(resume):
        print(f"Resuming from checkpoint: {resume}")
        trainer.train(resume_from_checkpoint=resume)
    else:
        print("Starting training...")
        trainer.train()

    final_path = os.path.join(config.output_dir, "final_model")
    
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    print(f"Final model saved -> {final_path}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Fine-tune maya1 TTS model")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config) if os.path.exists(args.config) else TrainConfig()
    train(cfg)