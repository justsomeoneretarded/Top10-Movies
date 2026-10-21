from __future__ import annotations

import gc
import os
import re
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
        token = key("voice-v2-padded-boundaries", text, self.reference_hash, self.model_name)
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
            # Keep the first/last phonemes clear of encoder and edit boundaries.
            # Padding adds real silence; it never trims or shifts speech out of the file.
            ffmpeg("-i", raw, "-af",
                   "asetpts=PTS-STARTPTS,adelay=160:all=1,apad=pad_dur=0.26,loudnorm=I=-18:TP=-2:LRA=9",
                   "-ar", "48000", "-ac", "1", output)
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


_NUMBER_WORDS = dict(zip(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"],
    range(20),
))
_NUMBER_WORDS.update(dict(zip(["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"], range(20, 100, 10))))


def _number_value(words: list[str]) -> int:
    """Normalize ordinary spoken numbers and common split-year readings."""
    values = [_NUMBER_WORDS.get(word, int(word) if word.isdigit() else 0) for word in words]
    if len(words) >= 2 and values[0] in (18, 19, 20) and words[1] not in ("hundred", "thousand"):
        remainder = _number_value(words[1:])
        if 10 <= remainder < 100:
            return values[0]*100 + remainder
    total = current = 0
    for word, value in zip(words, values):
        if word == "hundred":
            current = max(1, current)*100
        elif word == "thousand":
            total += max(1, current)*1000
            current = 0
        else:
            current += value
    return total + current


def _speech_tokens(text: str) -> list[str]:
    """Ignore punctuation and tolerate ASR contraction/number formatting changes."""
    text = text.casefold().replace("’", "'")
    for contraction, expanded in {
        "can't": "can not", "cannot": "can not", "won't": "will not", "let's": "let us",
        "i'm": "i am", "it's": "it is", "that's": "that is", "there's": "there is",
        "here's": "here is", "what's": "what is", "who's": "who is", "he's": "he is", "she's": "she is",
    }.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", expanded, text)
    for ending, expanded in (("n't", " not"), ("'re", " are"), ("'ve", " have"), ("'ll", " will"), ("'d", " would")):
        text = re.sub(r"\b(\w+)" + re.escape(ending) + r"\b", r"\1" + expanded, text)
    words = re.findall(r"\w+", text)
    tokens = []
    index = 0
    while index < len(words):
        start = index
        numeric = []
        while index < len(words):
            word = words[index]
            if word in _NUMBER_WORDS or word in ("hundred", "thousand") or word.isdigit():
                numeric.append(word)
                index += 1
            elif numeric and word == "and" and index+1 < len(words) and words[index+1] in _NUMBER_WORDS:
                index += 1
            else:
                break
        if numeric:
            tokens.append(str(_number_value(numeric)))
        else:
            tokens.append(words[start])
            index += 1
    return tokens


def speech_verified(expected: str, cues: list[dict]) -> bool:
    """Check overall speech and both ends without requiring word-perfect ASR.

    A long paragraph can retain 72% coverage after losing an entire opening or
    closing sentence. Require evidence near both ends as well. Up to two edge
    tokens and scattered recognition errors remain tolerable.
    """
    from difflib import SequenceMatcher

    planned = _speech_tokens(expected)
    heard = _speech_tokens(" ".join(cue.get("text", "") for cue in cues))
    if not planned or not heard:
        return False
    blocks = SequenceMatcher(None, planned, heard, autojunk=False).get_matching_blocks()
    matched = {index for block in blocks for index in range(block.a, block.a+block.size)}
    if len(matched)/len(planned) < .72:
        return False
    edge_allowance = min(2, max(1, len(planned)//8))
    if min(matched) > edge_allowance or len(planned)-1-max(matched) > edge_allowance:
        return False
    window = min(8, len(planned))
    required = max(1, (window+1)//2)
    return (sum(index in matched for index in range(window)) >= required and
            sum(index in matched for index in range(len(planned)-window, len(planned))) >= required)
