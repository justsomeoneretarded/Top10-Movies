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

    def speak(self, text: str, attempt_offset: int = 0, *, force: bool = False) -> Path:
        import numpy as np
        import soundfile as sf
        import torch
        baseline = key("voice-v3-clean-presence", text, self.reference_hash, self.model_name)
        token = key("voice-retry-v1", baseline, attempt_offset) if force else baseline
        output = self.cache / f"{token}.wav"
        suffix = 0
        if not force and output.exists() and duration(output) > 0.5:
            return output
        if force:
            # A later use of the same text must not overwrite already-verified
            # audio or captions belonging to another paragraph.
            while output.exists() or output.with_suffix(".captions.json").exists():
                suffix += 1
                output = self.cache / f"{token}-{suffix}.wav"
        for attempt in range(3):
            torch.manual_seed(int(baseline[:8], 16) + attempt + attempt_offset + suffix*1009)
            with torch.inference_mode():
                wave = self.model.generate(text)
            raw = output.with_suffix(".raw.wav")
            samples = wave.squeeze().detach().cpu().numpy()
            if not np.isfinite(samples).all() or not samples.size or np.max(np.abs(samples)) < 1e-5:
                print(f"Voice attempt {attempt+1}: empty, silent, or invalid audio; retrying.", flush=True)
                continue
            sf.write(raw, samples, self.model.sr, subtype="PCM_24")
            # Keep the first/last phonemes clear of encoder and edit boundaries.
            # Padding adds real silence; it never trims or shifts speech out of the file.
            ffmpeg("-i", raw, "-af",
                   "asetpts=PTS-STARTPTS,highpass=f=65,agate=threshold=0.001:ratio=1.5:range=0.25:attack=5:release=180:knee=4,equalizer=f=3000:t=q:w=0.8:g=1.5,adelay=160:all=1,apad=pad_dur=0.26,loudnorm=I=-18:TP=-2:LRA=9",
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


def captions(audio: Path, model, *, vad_filter: bool = True, beam_size: int = 3,
             condition_on_previous_text: bool = False) -> list[dict]:
    segments, _ = model.transcribe(str(audio), language="en", beam_size=beam_size, word_timestamps=True,
                                 vad_filter=vad_filter, condition_on_previous_text=condition_on_previous_text)
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


def _under_hundred(words: list[str]) -> int | None:
    values = [_NUMBER_WORDS.get(word, int(word) if word.isdigit() else -1) for word in words]
    if len(values) == 1 and 0 <= values[0] < 100:
        return values[0]
    if len(values) == 2 and values[0] in range(20, 100, 10) and 0 < values[1] < 10:
        return sum(values)
    return None


def _under_thousand(words: list[str]) -> int | None:
    if len(words) >= 2 and words[1] == "hundred":
        prefix = _under_hundred(words[:1])
        suffix = _under_hundred(words[2:]) if len(words) > 2 else 0
        if prefix is not None and prefix > 0 and suffix is not None:
            return prefix*100 + suffix
    return _under_hundred(words)


def _number_value(words: list[str]) -> int | None:
    """Parse valid number phrases; adjacent independent numbers never get added."""
    words = [word for word in words if word != "and"]
    if len(words) == 1:
        if words[0].isdigit():
            return int(words[0])
        return {"hundred": 100, "thousand": 1000}.get(words[0], _NUMBER_WORDS.get(words[0]))
    first = _under_hundred(words[:1])
    if first in (18, 19, 20):
        remainder = _under_hundred(words[1:])
        if remainder is not None and 10 <= remainder < 100:
            return first*100 + remainder
    if words.count("thousand") == 1:
        split = words.index("thousand")
        prefix = _under_thousand(words[:split])
        suffix = _under_thousand(words[split+1:]) if split+1 < len(words) else 0
        if prefix is not None and prefix > 0 and suffix is not None:
            return prefix*1000 + suffix
    return _under_thousand(words)


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
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    # Sentence/clause punctuation ends a number phrase. Hyphens still permit
    # ordinary compounds such as twenty-six.
    words = re.findall(r"\w+|[.!?;:,\n]", text)
    tokens = []
    index = 0
    while index < len(words):
        if not re.fullmatch(r"\w+", words[index]):
            index += 1
            continue
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
            # Partition a numeric run into valid phrases, preserving lists such
            # as one two three instead of treating them as the number six.
            while numeric:
                for stop in range(len(numeric), 0, -1):
                    value = _number_value(numeric[:stop])
                    if value is not None:
                        tokens.append(str(value))
                        numeric = numeric[stop:]
                        break
                else:
                    tokens.append(numeric.pop(0))
        else:
            tokens.append(words[start])
            index += 1
    return tokens


def _number_spelling(value: int) -> str:
    small = list(_NUMBER_WORDS)
    if value < 20:
        return small[value]
    if value < 100:
        return small[18+value//10] + (_number_spelling(value % 10) if value % 10 else "")
    if value < 1000:
        return _number_spelling(value//100) + "hundred" + (_number_spelling(value % 100) if value % 100 else "")
    if value < 10000:
        return _number_spelling(value//1000) + "thousand" + (_number_spelling(value % 1000) if value % 1000 else "")
    return str(value)


def _aligned_tokens(planned: list[str], heard: list[str]) -> tuple[set[int], set[int]]:
    """Monotone word alignment with at most a three-token spelling join/split."""
    def spelling(token):
        return _number_spelling(int(token)) if token.isdigit() else token

    def spans(tokens):
        return {(start, size): "".join(spelling(word) for word in tokens[start:start+size])
                for start in range(len(tokens)) for size in range(1, min(3, len(tokens)-start)+1)}

    left, right = spans(planned), spans(heard)
    scores = [[(0, 0) for _ in range(len(heard)+1)] for _ in range(len(planned)+1)]
    trace = {}

    def update(i, j, score, step):
        if score > scores[i][j]:
            scores[i][j] = score
            trace[(i, j)] = step

    for i in range(len(planned)+1):
        for j in range(len(heard)+1):
            score = scores[i][j]
            if i < len(planned):
                update(i+1, j, score, (i, j, 0, 0))
            if j < len(heard):
                update(i, j+1, score, (i, j, 0, 0))
            for a, b in ((1, 1), (1, 2), (1, 3), (2, 1), (3, 1)):
                if i+a <= len(planned) and j+b <= len(heard) and left[i, a] == right[j, b]:
                    update(i+a, j+b, (score[0]+a, score[1]+b), (i, j, a, b))
    expected_matches, heard_matches = set(), set()
    position = (len(planned), len(heard))
    while position in trace:
        i, j, a, b = trace[position]
        expected_matches.update(range(i, i+a))
        heard_matches.update(range(j, j+b))
        position = (i, j)
    return expected_matches, heard_matches


def speech_check(expected: str, cues: list[dict]) -> dict:
    """Report completeness evidence without requiring word-perfect ASR.

    The original coverage and boundary requirements are retained. Bounded
    spelling joins tolerate ASR word segmentation; transcript precision rejects
    substantial extra/repeated speech instead of accepting a complete sentence
    merely because it appears somewhere inside a runaway recording.
    """
    planned = _speech_tokens(expected)
    transcript = " ".join(cue.get("text", "") for cue in cues)
    heard = _speech_tokens(transcript)
    absent_edge = {"accepted": False, "matched": 0, "required": max(1, (min(8, len(planned))+1)//2),
                   "unmatched_tokens": len(planned)}
    result = {"accepted": False, "coverage": 0.0, "precision": 0.0,
              "reason": "empty_expected" if not planned else "empty_transcript", "heard": transcript,
              "opening": dict(absent_edge), "ending": dict(absent_edge),
              "longest_unmatched_phrase": len(planned)}
    if not planned or not heard:
        return result
    matched, heard_matches = _aligned_tokens(planned, heard)
    result["coverage"] = len(matched)/len(planned)
    result["precision"] = len(heard_matches)/len(heard)
    longest_missing = current_missing = 0
    for index in range(len(planned)):
        current_missing = 0 if index in matched else current_missing+1
        longest_missing = max(longest_missing, current_missing)
    result["longest_unmatched_phrase"] = longest_missing
    edge_allowance = min(2, max(1, len(planned)//8))
    window = min(8, len(planned))
    required = max(1, (window+1)//2)
    for name, indices, omitted in (
        ("opening", range(window), min(matched) if matched else len(planned)),
        ("ending", range(len(planned)-window, len(planned)), len(planned)-1-max(matched) if matched else len(planned)),
    ):
        count = sum(index in matched for index in indices)
        result[name] = {"accepted": omitted <= edge_allowance and count >= required,
                        "matched": count, "required": required, "unmatched_tokens": omitted}
    if result["coverage"] < .72:
        result["reason"] = "coverage"
    elif not result["opening"]["accepted"]:
        result["reason"] = "opening"
    elif not result["ending"]["accepted"]:
        result["reason"] = "ending"
    elif longest_missing >= 6:
        result["reason"] = "missing_or_reordered_phrase"
    elif len(heard) > len(planned)+2 and result["precision"] < .72:
        result["reason"] = "extra_or_repeated_speech"
    else:
        result.update(accepted=True, reason="accepted")
    return result


def speech_verified(expected: str, cues: list[dict]) -> bool:
    """Compatibility wrapper for callers that only need the acceptance decision."""
    return speech_check(expected, cues)["accepted"]


def paced_take(source: Path, speed: float) -> Path:
    """Pitch-preserving narration pacing before edit and subtitle assembly."""
    from .render import _strict_audio_decode
    if speed == 1:
        return source
    output = source.with_name(source.stem+f"-pace-v3-{speed:.3f}.wav")
    if not output.exists():
        tempo = f"atempo={speed}"  # Preserve articulation without spectral/formant reconstruction.
        ffmpeg("-i", source, "-af", f"{tempo},asetpts=N/SR/TB",
               "-ar", "48000", "-ac", "1", "-c:a", "pcm_s24le", output)
    _strict_audio_decode(output)
    return output
