from __future__ import annotations

import json
import math
import re
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .core import PipelineError, duration, save_json


def analyse(path: Path, cache: Path) -> list[dict]:
    """Sample cuts and motion at 2fps; preserve source timestamps, excluding end slates."""
    cache = cache.with_name(cache.stem + ".calm-v2.json")
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
        for point in np.arange(start, end - 1.2, 8.0):
            stop = min(float(point + 8.0), end)
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
    """Soft CLIP relevance, stable shots, and occasional face holds; all CPU."""
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
        self.faces = {}
        self.people = {}
        self.held = set()

    def index(self, path: Path, shots: list[dict]):
        vectors = []
        face_flags = []
        face_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(str(face_path)) if face_path.exists() else None
        for offset in range(0, len(shots), 8):
            batch = shots[offset:offset+8]
            images = []
            for shot in batch:
                try:
                    image = get_frame(path, (shot["start"]+shot["end"])/2)
                except PipelineError:
                    image = Image.new("RGB", (320, 180), "gray")
                images.append(image)
                gray = cv2.cvtColor(np.array(image.resize((480, 270))), cv2.COLOR_RGB2GRAY)
                detected = detector.detectMultiScale(gray, 1.1, 4) if detector is not None else []
                face_flags.append(any(w*h >= 1000 for _, _, w, h in detected))
            inputs = self.processor(images=images, return_tensors="pt")
            with self.torch.inference_mode():
                values = self.model.get_image_features(**inputs)
                if not isinstance(values, self.torch.Tensor):
                    values = values.pooler_output
                vectors.append(values / values.norm(dim=-1, keepdim=True))
        self.features[str(path)] = self.torch.cat(vectors)
        self.faces[str(path)] = face_flags

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
        try:
            relevance = self.relevance(path, prompt)
        except (RuntimeError, ValueError, KeyError):
            relevance = np.zeros(len(shots))
        return max(float(relevance[i]) + (min(s["motion"], 40)/200 if action else 0) for i, s in enumerate(shots))

    def choose(self, path: Path, shots: list[dict], text: str, seconds: float, action: bool, maximum_shot: float | None = None) -> list[dict]:
        if not shots or seconds <= 0:
            return []
        # Similarity is a soft preference only: weak matches never reject a shot or a video.
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()] or [text]
        weights = np.cumsum([max(1, len(part.split())) for part in sentences])
        remaining = seconds
        chosen = []
        last_index = self.last_shots.get(str(path), -1)
        maximum_shot = maximum_shot or (7.5 if action else 8.5)
        actor_names = getattr(self, "people", {}).get(str(path), [])
        about_person = bool(re.search(r"\b(actor|actress|performance|plays|portrays|starring|character)\b", text, re.IGNORECASE)) or any(name.casefold() in text.casefold() for name in actor_names)
        flags = getattr(self, "faces", {}).get(str(path), [False]*len(shots))
        held = getattr(self, "held", set())
        self.held = held
        relevance_cache = {}
        while remaining > 0.01:
            position = min(len(sentences)-1, int(np.searchsorted(weights, (seconds-remaining)/seconds * weights[-1], side="right")))
            if position not in relevance_cache:
                try:
                    relevance_cache[position] = self.relevance(path, sentences[position])
                except (RuntimeError, ValueError, KeyError):
                    relevance_cache[position] = np.zeros(len(shots))
            relevance = relevance_cache[position]
            scores = []
            for index, shot in enumerate(shots):
                used = self.uses.get((str(path), index), 0)
                available = shot["end"]-shot["start"]
                score = float(relevance[index]) - used*.04
                # Stability and readable source shots matter more than high motion.
                score += min(available, maximum_shot)*.035 - min(shot["motion"], 60)/600
                if about_person and str(path) not in held and flags[index]:
                    score += .22
                if index == last_index:
                    score -= .35
                elif last_index >= 0 and abs(shot["start"]-shots[last_index]["end"]) < .65:
                    score -= .08
                scores.append(score)
            index = int(np.argmax(np.nan_to_num(scores, nan=-1.0)))
            shot = shots[index]
            available = max(.1, shot["end"]-shot["start"])
            portrait = about_person and str(path) not in held and flags[index] and remaining >= 3
            # A face discussion gets a readable still; short source edits get breathing room.
            freeze = 0 if portrait else max(0, min(4.5, remaining)-available)
            length = min(remaining, 3.2 if portrait else available+freeze, maximum_shot)
            if 0 < remaining-length < 2.5:
                # Keep the final thought on this image rather than flashing a tiny extra cut.
                freeze += remaining-length
                length = remaining
            freeze = min(freeze, max(0, length-.1))
            chosen.append({"start": (shot["start"]+shot["end"])/2 if portrait else shot["start"], "duration": length, "freeze": freeze, "still": bool(portrait),
                           "visual_reason": "portrait-hold" if portrait else "loose-script-match"})
            if portrait:
                held.add(str(path))
            self.cut_count += 1
            self.uses[(str(path), index)] = self.uses.get((str(path), index), 0)+1
            remaining -= length
            last_index = index
        self.last_shots[str(path)] = last_index
        return chosen


def intro_montage(selector, trailers: list[Path], shot_sets: list[list[dict]], paragraph_seconds: list[float], brief: str, action: bool) -> list[list[dict]]:
    """Readable multi-title hook, reducing coverage when narration is too short."""
    count = len(trailers)
    required = min(math.ceil(count/2), max(1, int(sum(paragraph_seconds)/2.5)))
    frame_counts = [math.ceil(seconds*30) for seconds in paragraph_seconds]
    if not count or not frame_counts or sum(frame_counts) < required:
        raise PipelineError("Intro audio is too short to show the required title coverage.")
    # Give each hook image roughly three seconds, retaining feasible multi-title coverage.
    slot_frames = 90 if action else 105
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
        print("Short intro uses fewer titles to preserve readable pacing.", flush=True)
    return plans
