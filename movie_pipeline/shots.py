from __future__ import annotations

import json
import math
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .core import PipelineError, duration, save_json


def analyse(path: Path, cache: Path) -> list[dict]:
    """Sample cuts and motion at 2fps; preserve source timestamps, excluding end slates."""
    if cache.exists():
        return json.loads(cache.read_text())
    cap = cv2.VideoCapture(str(path))
    length = duration(path)
    previous = None
    boundaries = [max(2.0, length * 0.04)]
    samples = []
    for tick in np.arange(boundaries[0], length * 0.90, 0.5):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(tick) * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
        motion = float(np.abs(gray.astype(float) - previous.astype(float)).mean()) if previous is not None else 0
        if motion > 38 and tick - boundaries[-1] >= 1.5:
            boundaries.append(float(tick))
        samples.append((float(tick), motion, float(gray.mean()), float(gray.std())))
        previous = gray
    boundaries.append(length * 0.90)
    cap.release()
    shots = []
    for start, end in pairwise(boundaries):
        # Long uncut shots are subdivided to provide useful choices.
        for point in np.arange(start, end - 1.2, 4.5):
            stop = min(float(point + 4.5), end)
            metrics = [s for s in samples if point <= s[0] < stop]
            if not metrics or np.mean([s[2] for s in metrics]) < 15 or np.mean([s[3] for s in metrics]) < 12:
                continue
            shots.append({"start": float(point), "end": stop, "motion": float(np.mean([s[1] for s in metrics]))})
    if not shots:
        raise PipelineError(f"No usable shots detected in {path.name}.")
    save_json(cache, shots)
    return shots


def get_frame(path: Path, seconds: float) -> Image.Image:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise PipelineError(f"Cannot read frame from {path.name} at {seconds:.1f}s.")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


class ShotSelector:
    """CLIP relevance plus motion preference and reuse penalties; all CPU."""
    def __init__(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.features = {}
        self.uses = {}
        self.last_shots = {}
        self.cut_count = 0

    def index(self, path: Path, shots: list[dict]):
        vectors = []
        for offset in range(0, len(shots), 8):
            batch = shots[offset:offset+8]
            images = [get_frame(path, (s["start"]+s["end"])/2) for s in batch]
            inputs = self.processor(images=images, return_tensors="pt")
            with self.torch.inference_mode():
                values = self.model.get_image_features(**inputs)
                if not isinstance(values, self.torch.Tensor):
                    values = values.pooler_output
                vectors.append(values / values.norm(dim=-1, keepdim=True))
        self.features[str(path)] = self.torch.cat(vectors)

    def relevance(self, path: Path, text: str):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77)
        with self.torch.inference_mode():
            vector = self.model.get_text_features(**inputs)
            if not isinstance(vector, self.torch.Tensor):
                vector = vector.pooler_output
            vector = vector / vector.norm(dim=-1, keepdim=True)
            relevance = (self.features[str(path)] @ vector.T).squeeze(-1).numpy()
        return relevance

    def hook_score(self, path: Path, shots: list[dict], prompt: str, action: bool) -> float:
        relevance = self.relevance(path, prompt)
        return max(float(relevance[i]) + (min(s["motion"], 40)/200 if action else 0) for i, s in enumerate(shots))

    def choose(self, path: Path, shots: list[dict], text: str, seconds: float, action: bool, maximum_shot: float | None = None) -> list[dict]:
        relevance = self.relevance(path, text)
        remaining = seconds
        chosen = []
        last_index = self.last_shots.get(str(path), -1)
        maximum_shot = maximum_shot or (3.4 if action else 4.2)
        while remaining > 0.01:
            scores = []
            for index, shot in enumerate(shots):
                used = self.uses.get((str(path), index), 0)
                score = float(relevance[index]) + (min(shot["motion"], 40)/400 if action else 0) - used * 0.055
                if index == last_index:
                    score -= 1
                elif last_index >= 0 and abs(shot["start"] - shots[last_index]["end"]) < 0.65:
                    # Avoid reconstructing the original trailer sequence across cuts/paragraphs.
                    score -= 0.35
                scores.append(score)
            index = int(np.argmax(scores))
            shot = shots[index]
            length = min(remaining, shot["end"] - shot["start"], maximum_shot)
            # Merge an extremely short tail into the preceding shot rather than flash one frame.
            if remaining - length < 0.5 and remaining <= shot["end"] - shot["start"]:
                length = remaining
            # Short holds emphasize occasional visual beats without stopping narration.
            freeze = min(0.22 if action else 0.3, length/5) if self.cut_count % 7 == 0 and length >= 0.9 else 0
            chosen.append({"start": shot["start"], "duration": length, "freeze": freeze})
            self.cut_count += 1
            self.uses[(str(path), index)] = self.uses.get((str(path), index), 0) + 1
            remaining -= length
            last_index = index
        self.last_shots[str(path)] = last_index
        return chosen


def intro_montage(selector, trailers: list[Path], shot_sets: list[list[dict]], paragraph_seconds: list[float], brief: str, action: bool) -> list[list[dict]]:
    """Round-robin hook montage covers at least ceil(N/2) titles across the full intro."""
    count = len(trailers)
    required = math.ceil(count/2)
    frame_counts = [math.ceil(seconds*30) for seconds in paragraph_seconds]
    if not count or not frame_counts or sum(frame_counts) < required:
        raise PipelineError("Intro audio is too short to show the required title coverage.")
    # Distribute cuts around 1–2 seconds; add cuts if needed to meet title coverage.
    slot_frames = 45 if action else 60
    slots = [max(1, math.ceil(frames/slot_frames)) for frames in frame_counts]
    while sum(slots) < required:
        options = [i for i, frames in enumerate(frame_counts) if slots[i] < frames]
        i = max(options, key=lambda i: frame_counts[i]/slots[i])
        slots[i] += 1
    prompt = ("A cinematic action highlight: a fight, chase, stunt, explosion, or dramatic physical action. " if action else "A striking cinematic visual with a compelling character or dramatic environment. ") + brief
    ranked = sorted(range(count), key=lambda i: selector.hook_score(trailers[i], shot_sets[i], prompt, action), reverse=True)
    active = ranked[:min(count, sum(slots))]
    plans = []
    cursor = 0
    for frames, slot_count in zip(frame_counts, slots):
        plan = []
        quotient, remainder = divmod(frames, slot_count)
        for slot in range(slot_count):
            index = active[cursor % len(active)]
            cursor += 1
            seconds = (quotient + (slot < remainder))/30
            cuts = selector.choose(trailers[index], shot_sets[index], prompt, seconds, action, maximum_shot=seconds)
            plan.extend({**cut, "source": str(trailers[index]), "film_index": index} for cut in cuts)
        plans.append(plan)
    covered = {shot["film_index"] for plan in plans for shot in plan}
    if len(covered) < required:
        raise PipelineError("Intro montage did not cover at least half of the supplied titles.")
    return plans
