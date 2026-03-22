import torch
import torchaudio
from typing import List, Optional

from snac import SNAC

from .constants import SNAC_MODEL_NAME, SNAC_SAMPLE_RATE,CODE_TOKEN_OFFSET


class SNACEncoder:

    def __init__(self, device: str = "cuda"):
        
        self.device = device
        
        print(f"Loading {SNAC_MODEL_NAME} on {device}")
        
        self.model = SNAC.from_pretrained(SNAC_MODEL_NAME).eval().to(device)

    @torch.inference_mode()
    def encode_audio(self, wav_path: str) -> Optional[torch.Tensor]:

        try:
            wav, sr = torchaudio.load(wav_path)

            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            if sr != SNAC_SAMPLE_RATE:
                wav = torchaudio.transforms.Resample(sr, SNAC_SAMPLE_RATE)(wav)

            wav = wav.to(self.device)
            codes = self.model.encode(wav.unsqueeze(0))
            
            return self._pack_snac_to_7(codes).cpu()

        except Exception as e:
            print(f"encode_audio failed for '{wav_path}': {e}")
            return None

    def _pack_snac_to_7(self, codes: List[torch.Tensor]) -> torch.Tensor:

        l1 = codes[0].squeeze(0)
        l2 = codes[1].squeeze(0)
        l3 = codes[2].squeeze(0)

        frames = l1.shape[0]
        packed = []

        for i in range(frames):
            frame = [
                l1[i].item(),
                l2[2 * i].item(),
                l3[4 * i + 0].item(),
                l3[4 * i + 1].item(),
                l2[2 * i + 1].item(),
                l3[4 * i + 2].item(),
                l3[4 * i + 3].item(),
            ]
            packed.extend(CODE_TOKEN_OFFSET + c for c in frame)

        return torch.tensor(packed, dtype=torch.long)