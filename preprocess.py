import os
import json
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from maya.config import TrainConfig, load_config
from maya.snac_encoder import SNACEncoder
from maya.constants import SNAC_TOKENS_PER_FRAME


def _load_metadata(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"metadata_final.json must be a JSON array, got {type(data)}")
    return data


def _ensure_description(raw_text: str, default: str) -> str:
    if '<description="' in raw_text:
        return raw_text
    return f'<description="{default}"> {raw_text}'



def preprocess(config: TrainConfig):
    
    os.makedirs(config.preprocessed_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Reading metadata from '{config.metadata_path}'")
    metadata = _load_metadata(config.metadata_path)
    print(f"{len(metadata)} entries found.")

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    snac_encoder = SNACEncoder(device=device)

    success, skipped, error = 0, 0, 0

    for item in tqdm(metadata, desc="Preprocessing"):
        
        file_id = item.get("id")
        
        if not file_id:
            skipped += 1
            continue

        save_path = os.path.join(config.preprocessed_dir, f"{file_id}.pt")
        audio_path = os.path.join(config.dataset_dir, f"{file_id}.wav")

        if os.path.exists(save_path):
            success += 1
            continue

        if not os.path.exists(audio_path):
            print(f"Missing audio: {audio_path}")
            skipped += 1
            continue

        try:

            raw_text = item.get("formatted_text", "")
            full_text = _ensure_description(raw_text, config.default_description)

            text_tokens = tokenizer(
                full_text,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"].squeeze(0)

            if text_tokens.shape[0] > config.max_text_len:
                text_tokens = text_tokens[: config.max_text_len]

            snac_tokens = snac_encoder.encode_audio(audio_path)

            if snac_tokens is None or len(snac_tokens) < 7:
                print(f"SNAC encode failed or too short: {audio_path}")
                skipped += 1
                continue

            if snac_tokens.shape[0] > config.max_speech_len:
                limit = (config.max_speech_len // SNAC_TOKENS_PER_FRAME) * SNAC_TOKENS_PER_FRAME
                snac_tokens = snac_tokens[:limit]

            torch.save(
                {
                    "text_tokens": text_tokens, 
                    "snac_tokens": snac_tokens, 
                    "file_id": file_id
                },
                save_path,
            )
            success += 1

        except Exception as e:
            print(f"Error on '{file_id}': {e}")
            error += 1

    print(f"\nDone -> success: {success}, skipped: {skipped}, errors: {error}")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Preprocess maya1 TTS dataset")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config) if os.path.exists(args.config) else TrainConfig()
    preprocess(cfg)