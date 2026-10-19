from __future__ import annotations

import gc
import os
from pathlib import Path

from .core import PipelineError, duration, ffmpeg, key


class Narrator:
    def __init__(self, reference: Path, cache: Path):
        import torch
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        torch.set_num_threads(int(os.getenv("TTS_THREADS", str(os.cpu_count() or 2))))
        self.model_name = os.getenv("TTS_MODEL", "turbo")
        if self.model_name not in ("turbo", "nano"):
            raise PipelineError("TTS_MODEL must be turbo or nano.")
        self.model = ChatterboxTurboTTS.from_pretrained(device="cpu", nano=self.model_name == "nano")
        self.model.prepare_conditionals(str(reference))
        self.reference_hash = key(reference.read_bytes().hex())
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)

    def speak(self, text: str, attempt_offset: int = 0) -> Path:
        import numpy as np
        import soundfile as sf
        import torch
        token = key("voice-v1", text, self.reference_hash, self.model_name)
        output = self.cache / f"{token}.wav"
        if output.exists() and duration(output) > 0.5:
            return output
        for attempt in range(3):
            torch.manual_seed(int(token[:8], 16) + attempt + attempt_offset)
            with torch.inference_mode():
                wave = self.model.generate(text)
            raw = self.cache / f"{token}.raw.wav"
            samples = wave.squeeze().detach().cpu().numpy()
            if not np.isfinite(samples).all() or not samples.size or np.max(np.abs(samples)) < 1e-5:
                print(f"Voice attempt {attempt+1}: empty, silent, or invalid audio; retrying.", flush=True)
                continue
            sf.write(raw, samples, self.model.sr)
            ffmpeg("-i", raw, "-af", "loudnorm=I=-18:TP=-2:LRA=9", "-ar", "48000", "-ac", "1", output)
            raw.unlink(missing_ok=True)
            seconds = duration(output)
            # Rate is diagnostic only. Actual missing/repeated speech is checked by transcription.
            if 0.15 <= seconds <= max(30, len(text.split())*2+10):
                wpm = len(text.split()) * 60 / seconds
                print(f"Voice generated: {len(text.split())} words, {seconds:.1f}s, {wpm:.0f} words/min; transcript verification follows.", flush=True)
                return output
            print(f"Voice attempt {attempt+1}: unusable audio duration {seconds:.1f}s; retrying.", flush=True)
            output.unlink(missing_ok=True)
        raise PipelineError("Voice generation returned empty, silent, invalid, or runaway audio after three attempts. Speaking speed alone does not cause this failure.")

    def close(self):
        del self.model
        gc.collect()


def captions(audio: Path, model) -> list[dict]:
    segments, _ = model.transcribe(str(audio), language="en", beam_size=3, word_timestamps=True, vad_filter=True)
    result = []
    for segment in segments:
        words = segment.words or []
        chunk = []
        for word in words:
            chunk.append(word)
            if len(" ".join(w.word.strip() for w in chunk)) >= 55 or word.end - chunk[0].start >= 3.5:
                result.append({"start": chunk[0].start, "end": chunk[-1].end, "text": " ".join(w.word.strip() for w in chunk)})
                chunk = []
        if chunk:
            result.append({"start": chunk[0].start, "end": chunk[-1].end, "text": " ".join(w.word.strip() for w in chunk)})
        if not words and segment.text.strip():
            result.append({"start": segment.start, "end": segment.end, "text": segment.text.strip()})
    return result


def spoken_coverage(expected: str, cues: list[dict]) -> float:
    from difflib import SequenceMatcher

    from .core import normalized
    a = normalized(expected).split()
    b = normalized(" ".join(c["text"] for c in cues)).split()
    matched = sum(block.size for block in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    return matched / max(1, len(a))
