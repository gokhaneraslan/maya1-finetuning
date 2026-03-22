import os
import torch
import soundfile as sf
from transformers import TrainerCallback

from .constants import (
    PAD_TOKEN_ID, CODE_END_TOKEN_ID,
    SNAC_MIN_ID, SNAC_MAX_ID, SNAC_SAMPLE_RATE,
)
from .utils import build_prompt, lower_turkish, unpack_snac_from_7


class AudioSampleCallback(TrainerCallback):

    def __init__(
        self,
        output_dir: str,
        snac_model,
        tokenizer,
        sample_text: str,
        sample_description: str,
        device: str = "cuda",
    ):
        self.output_dir = output_dir
        self.snac_model = snac_model
        self.tokenizer = tokenizer
        self.sample_text = sample_text
        self.sample_description = sample_description
        self.device = device


    def on_save(self, args, state, control, model, **kwargs):
        model.eval()
        try:
            self._generate_and_save(model, state.global_step)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            model.train()


    @torch.inference_mode()
    def _generate_and_save(self, model, step: int):
        
        text   = lower_turkish(self.sample_text)
        prompt = build_prompt(self.tokenizer, self.sample_description, text)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        print(f"Generating sample at step {step}")

        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            min_new_tokens=28,
            temperature=0.4,
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
            print("Not enough tokens -> skipping.")
            return

        levels = unpack_snac_from_7(snac_tokens)
        codes  = [torch.tensor(l, dtype=torch.long, device=self.device).unsqueeze(0)for l in levels]

        z_q   = self.snac_model.quantizer.from_codes(codes)
        audio = self.snac_model.decoder(z_q)[0, 0].cpu().numpy()

        if len(audio) > 2048:
            audio = audio[2048:]

        out_dir  = os.path.join(self.output_dir, "output")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"sample_audio-{step}.wav")

        sf.write(out_path, audio, SNAC_SAMPLE_RATE)
        print(f"Saved -> {out_path}")