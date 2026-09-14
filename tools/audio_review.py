"""Reproducible external audio review; this model is an evaluator, not the composer.

Run in a separate GPU environment with transformers==4.57.6, accelerate,
librosa and soundfile. Keeps audio hashes, the pinned model revision and the
verbatim response. A single model's opinion is not a human listening study.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path


def main():
    import librosa
    import torch
    import transformers
    from huggingface_hub import model_info
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    p = argparse.ArgumentParser()
    p.add_argument("audio", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--brief", default="")
    p.add_argument("--model", default="Qwen/Qwen2.5-Omni-7B")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    revision = model_info(a.model).sha
    torch.set_num_threads(4)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        a.model,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        enable_audio_output=False,
        attn_implementation="sdpa",
    )
    model.disable_talker()
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(a.model, revision=revision)
    prompt = (
        "Listen to the whole recording and give a concise, candid musical review. "
        "Describe the audible instruments, pulse and phrasing, melodic development, "
        "harmonic direction, emotional character, contrasts, surprises and ending. "
        "Identify up to four concrete audible weaknesses, if present, and suggest "
        "musical changes. Distinguish composition from performance or sound quality. "
        "Do not assume a composer, genre, method, or quality in advance. "
        "Do not invent precise chords or timestamps if uncertain."
    )
    if a.brief:
        prompt += " Intended brief (assess fit rather than assume it): " + a.brief
    for path in a.audio:
        started = time.monotonic()
        audio, sr = librosa.load(path, sr=16000, mono=True)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(
            text=text,
            audio=[audio],
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(model.device).to(model.dtype)
        with torch.inference_mode():
            ids = model.generate(
                **inputs,
                return_audio=False,
                thinker_max_new_tokens=750,
                thinker_do_sample=False,
            )
        response = processor.batch_decode(
            ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )[0]
        receipt = {
            "model": a.model,
            "revision": revision,
            "transformers": transformers.__version__,
            "audio": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "seconds_of_audio": len(audio) / sr,
            "prompt": prompt,
            "response": response,
            "runtime_seconds": time.monotonic() - started,
            "limitation": "External model opinion, not a verified human perceptual judgment.",
        }
        name = path.parent.name + "-" + path.stem
        (a.out / (name + ".json")).write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
