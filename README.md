# Maya1 Turkish TTS — Full Fine-Tuning

Fine-tune **[Maya1](https://huggingface.co/maya-research/maya1)** on a Turkish (or any language) dataset using full fine-tuning.

---

## About Maya1

Maya1 is a state-of-the-art open-source Text-to-Speech model developed by **[Maya Research](https://mayaresearch.ai)**, backed by **South Park Commons**.

| Property | Value |
|---|---|
| **Architecture** | 3B-parameter Llama-style transformer + SNAC codec |
| **Model Type** | Text-to-Speech, Emotional Voice Synthesis, Voice Design |
| **Language** | English (multi-accent) — this repo adds Turkish |
| **Audio Quality** | 24 kHz mono, ~0.98 kbps streaming |
| **Developed by** | Maya Research |

Maya1 takes natural language voice descriptions and inline emotion tags and produces expressive, human-quality speech. It's the only open-source model offering 20+ emotions, zero-shot voice design, and production-ready streaming in a single package.

```
<description="Female, in her 30s, warm timbre, neutral tone"> Merhaba, bugün hava çok güzel.
```

Supported emotion tags: `<laugh>` `<cry>` `<whisper>` `<sigh>` `<gasp>` `<angry>` `<giggle>` `<chuckle>` and 12+ more.

---

## Requirements

- **GPU:** NVIDIA H100 (recommended) or min 48GB
- **VRAM:** 80GB for full fine-tuning (all parameters unfrozen + gradient checkpointing)
- **Disk:** 100GB+ free space
- **Python:** 3.10+

---

## 1. Installation

```bash
# System dependencies
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3-pip python3-venv ffmpeg unzip wget espeak-ng

git clone https://github.com/gokhaneraslan/maya1-finetuning

cd maya1-finetuning

# Python dependencies
pip install -r requirements.txt
```

---

## 2. Flash Attention 2 (Optional but Recommended)

Flash Attention 2 speeds up training ~3x on H100/A100. **Do not install via pip** — it must match your exact Python + PyTorch + CUDA version.

### Option A — Pre-compiled wheel (fastest, ~1 min)

A pre-compiled wheel for the following environment is included in this repo:

- Python: 3.12.3
- PyTorch: 2.8.0 (cu128)
- Architecture: Linux x86\_64

```bash
pip install -y flash-attn
pip install flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
python -c "import flash_attn; print('Flash Attention:', flash_attn.__version__)"
```

### Option B — Compile from source (~15 min)

Use this if your environment differs from the wheel above.

```bash
pip uninstall -y flash-attn

cd /workspace
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention

export MAX_JOBS=4
pip install ninja packaging
python setup.py install

cd ..
python -c "import flash_attn; print('Flash Attention:', flash_attn.__version__)"
```

> If you skip Flash Attention entirely, switch `attn_implementation` in `maya/model.py` and `inference.py` from `"flash_attention_2"` to `"sdpa"`.

---

## 3. Dataset Format

Maya1 is a large, expressive model — it needs a dataset that matches its training format exactly. If your data doesn't follow this structure, the model will produce silence or noise.

Each sample in `metadata_final.json` must follow this format:

```json
[
  {
    "id": "sample_0001",
    "formatted_text": "<description=\"Female, in her 30s, neutral tone, clear diction\"> Merhaba, bugün hava gerçekten çok güzel."
  },
  {
    "id": "sample_0002",
    "formatted_text": "<description=\"Male, warm timbre, slow pacing, calm tone\"> Sizi bu akşam aramızda görmekten büyük mutluluk duyuyoruz."
  }
]
```

And the corresponding audio files must be at:

```
data/
  wavs/
    sample_0001.wav
    sample_0002.wav
    ...
  metadata_final.json
```

### Audio requirements

| Property | Value |
|---|---|
| Format | WAV (mono) |
| Sample rate | 24 kHz (resampled automatically if different) |
| Duration | 1–14 seconds per clip recommended |
| Normalization | -23 LUFS recommended |

### Description field guidelines

The `<description="...">` tag is the voice design prompt. It should describe the speaker naturally, as if briefing a voice actor. Include as many of the following as relevant:

- Gender and approximate age — `"Female, in her 30s"`
- Accent — `"Turkish accent"`, `"American accent"`
- Pitch — `"deep pitch"`, `"high pitch"`, `"normal pitch"`
- Timbre — `"warm timbre"`, `"gravelly timbre"`, `"clear voice"`
- Pacing — `"slow pacing"`, `"conversational pacing"`, `"fast delivery"`
- Tone — `"neutral tone"`, `"happy tone"`, `"sad tone"`, `"angry tone"`
- Role — `"narrator"`, `"event host"`, `"news anchor"`

**Minimum viable description:**
```
"<description=\"Female, neutral tone\"> text here"
```

**Rich description (better results):**
```
"<description=\"Realistic female voice in her 30s with a Turkish accent. Deep pitch, warm timbre, conversational pacing, neutral tone, narrator role.\"> text here"
```

Samples without a `<description="...">` tag will get the default from `config.yaml` (`default_description`). This works but produces less consistent voice style across the dataset.

---

## 4. Configuration

Edit `config.yaml` before running anything. The most important fields:

```yaml
# Paths — update these to match your setup
model_path: "maya-research/maya1"
dataset_dir: "./data/wavs"
metadata_path: "./data/metadata_final.json"
preprocessed_dir: "./data/preprocessed"
output_dir: "./output/maya1_finetune"

# Training
batch_size: 4
gradient_accumulation_steps: 16   # effective batch = 64
num_epochs: 100
learning_rate: 5e-5

sample_text: "Merhaba, bugün hava gerçekten çok güzel değil mi?"
sample_description: "Realistic female voice in her 30s with a Turkish accent. Warm timbre, neutral tone."
```

---

## 5. Preprocessing

Tokenises text and encodes audio into `.pt` files. Run once before training.

```bash
python preprocess.py
# or with a custom config:
python preprocess.py --config config.yaml
```

Preprocessed files are saved to `preprocessed_dir`. On subsequent runs, already-processed files are skipped automatically.

---

## 6. Training

```bash
python train.py
# or with a custom config:
python train.py --config config.yaml
```

To resume from a checkpoint, set `resume_from_checkpoint` in `config.yaml`:

```yaml
resume_from_checkpoint: "./output/maya1_finetune/checkpoint-1120"
```

### Monitoring

```bash
tensorboard --logdir ./output/maya1_finetune
```

A sample `.wav` is generated at every checkpoint save under `output/maya1_finetune/output/sample_audio-{step}.wav` — listen to these to track voice quality over training.

---

## 7. Inference

```bash
python inference.py \
    --checkpoint ./output/maya1_finetune/final_model \
    --text "Sesimi duyurabilmem epey uzun zaman aldı." \
    --desc "Female, in her 30s, Turkish accent, warm timbre, neutral tone" \
    --out output.wav
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to fine-tuned model directory |
| `--text` | required | Text to synthesise |
| `--desc` | built-in default | Voice description prompt |
| `--out` | `output.wav` | Output WAV file |
| `--temp` | `0.5` | Sampling temperature (0.1–1.0) |
| `--top_p` | `0.9` | Nucleus sampling |
| `--max_tokens` | `2048` | Max new tokens to generate |

---

## Project Structure

```
maya1-finetune/
├── config.yaml          # All settings — edit this before running
├── preprocess.py        # Tokenise text + encode audio → .pt files
├── train.py             # Full fine-tuning entry point
├── inference.py         # Generate speech from a checkpoint
├── requirements.txt
└── pre_build_flash_attn/
    ├── flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
└── maya/
    ├── __init__.py
    ├── constants.py     # Token IDs and SNAC constants
    ├── config.py        # TrainConfig dataclass + YAML loader
    ├── dataset.py       # Maya1Dataset + data_collator
    ├── snac_encoder.py  # Audio -> SNAC token encoding
    ├── model.py         # Model + tokenizer setup
    ├── callback.py      # AudioSampleCallback
    └── utils.py         # lower_turkish, build_prompt, unpack_snac
```

---

## Tips

- **First run:** preprocessing can take a while depending on dataset size. The `data/preprocessed/` folder caches results — safe to interrupt and resume.
- **VRAM:** if you run out, reduce `batch_size` to `2` and increase `gradient_accumulation_steps` to `32` to keep the effective batch size the same.
- **Quality vs speed:** lower `learning_rate` (e.g. `2e-5`) is safer for full fine-tuning but needs more epochs. `5e-5` trains faster but can overfit on small datasets.
- **Small dataset (<50h audio):** reduce `num_epochs` and monitor the sample wavs — once the voice sounds stable, stop early.