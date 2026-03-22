from typing import List
from transformers import PreTrainedTokenizer

from .constants import (
    SOH_ID, EOH_ID, SOA_ID, TEXT_EOT_ID,
    CODE_START_TOKEN_ID, CODE_END_TOKEN_ID, CODE_TOKEN_OFFSET,
)


def lower_turkish(text: str) -> str:
    """lowercase for Turkish (İ->i, I->ı)."""
    return text.replace("İ", "i").replace("I", "ı").lower()


def build_prompt(tokenizer: PreTrainedTokenizer, description: str, text: str) -> str:

    soh = tokenizer.decode([SOH_ID])
    eoh = tokenizer.decode([EOH_ID])
    soa = tokenizer.decode([SOA_ID])
    sos = tokenizer.decode([CODE_START_TOKEN_ID])
    eot = tokenizer.decode([TEXT_EOT_ID])
    bos = tokenizer.bos_token

    if not description:
        description = "neutral narration"

    formatted = f'<description="{description}"> {text}'
    
    return soh + bos + formatted + eot + eoh + soa + sos


def unpack_snac_from_7(snac_tokens: List[int]) -> List[List[int]]:

    if snac_tokens and snac_tokens[-1] == CODE_END_TOKEN_ID:
        snac_tokens = snac_tokens[:-1]

    frames = len(snac_tokens) // 7
    snac_tokens = snac_tokens[:frames * 7]

    if frames == 0:
        return [[], [], []]

    l1, l2, l3 = [], [], []

    for i in range(frames):
        slots = snac_tokens[i * 7 : i * 7 + 7]
        l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
        l2.extend([
            (slots[1] - CODE_TOKEN_OFFSET) % 4096,
            (slots[4] - CODE_TOKEN_OFFSET) % 4096,
        ])
        l3.extend([
            (slots[2] - CODE_TOKEN_OFFSET) % 4096,
            (slots[3] - CODE_TOKEN_OFFSET) % 4096,
            (slots[5] - CODE_TOKEN_OFFSET) % 4096,
            (slots[6] - CODE_TOKEN_OFFSET) % 4096,
        ])

    return [l1, l2, l3]