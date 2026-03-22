import sys
import argparse

import torch
import soundfile as sf
from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer

from maya.constants import (
    PAD_TOKEN_ID, CODE_END_TOKEN_ID,
    SNAC_MIN_ID, SNAC_MAX_ID,
    SNAC_SAMPLE_RATE, SNAC_MODEL_NAME,
)
from maya.utils import lower_turkish, build_prompt, unpack_snac_from_7


DEFAULT_DESCRIPTION = (
    "Realistic female voice in her 30s with a Turkish accent. "
    "Deep pitch, warm timbre, conversational pacing, neutral tone, narrator role."
)


class Maya1Inference:

    def __init__(
        self,
        checkpoint: str,
        base_model: str = None,
        device: str = None,
    ):
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        print(f"Loading model from '{checkpoint}'")
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
            #attn_implementation="flash_attention_2",
        ).eval()

        tok_path = base_model or checkpoint
        
        print(f"Loading tokenizer from '{tok_path}'")
        
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
        self.tokenizer.pad_token_id = PAD_TOKEN_ID

        print("Loading SNAC decoder...")
        self.snac_model = SNAC.from_pretrained(SNAC_MODEL_NAME).eval().to(self.device)

        print("Ready")


    @torch.inference_mode()
    def generate(
        self,
        text: str,
        description: str = DEFAULT_DESCRIPTION,
        temperature: float = 0.5,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ):

        clean = lower_turkish(text)
        
        print(f"Synthesising: \"{clean}\"")

        prompt = build_prompt(self.tokenizer, description, clean)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            min_new_tokens=28,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.2,
            do_sample=True,
            eos_token_id=CODE_END_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
        )

        generated_ids = outputs[0, inputs["input_ids"].shape[1]:].tolist()

        if CODE_END_TOKEN_ID in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(CODE_END_TOKEN_ID)]

        snac_tokens = [t for t in generated_ids if SNAC_MIN_ID <= t <= SNAC_MAX_ID]

        if len(snac_tokens) < 7:
            print("Not enough SNAC tokens generated.")
            return None

        levels = unpack_snac_from_7(snac_tokens)
        codes  = [torch.tensor(l, dtype=torch.long, device=self.device).unsqueeze(0)for l in levels]

        z_q   = self.snac_model.quantizer.from_codes(codes)
        audio = self.snac_model.decoder(z_q)[0, 0].cpu().numpy()

        if len(audio) > 2048:
            audio = audio[2048:]

        return audio


def main():
    
    parser = argparse.ArgumentParser(description="maya1 TTS inference")
    parser.add_argument("--checkpoint",  required=True,                 help="Fine-tuned model directory")
    parser.add_argument("--text",        required=True,                 help="Text to synthesise")
    parser.add_argument("--desc",        default=DEFAULT_DESCRIPTION,   help="Voice description prompt")
    parser.add_argument("--out",         default="output.wav",          help="Output WAV file path")
    parser.add_argument("--temp",        type=float, default=0.5,       help="Sampling temperature (0.1–1.0)")
    parser.add_argument("--top_p",       type=float, default=0.9,       help="Top-p nucleus sampling")
    parser.add_argument("--max_tokens",  type=int,   default=2048,      help="Max new tokens")
    parser.add_argument("--base_model",  default=None,                  help="Override tokenizer source (usually not needed)")
    args = parser.parse_args()

    tts   = Maya1Inference(checkpoint=args.checkpoint, base_model=args.base_model)
    
    audio = tts.generate(
        text=args.text,
        description=args.desc,
        temperature=args.temp,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    if audio is not None:
        sf.write(args.out, audio, SNAC_SAMPLE_RATE)
        print(f"Saved -> {args.out}")
    else:
        print("Generation failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()