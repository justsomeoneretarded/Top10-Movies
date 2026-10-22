from __future__ import annotations

import io
import json
import math
import re
import shutil
import subprocess
import unicodedata
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .core import PipelineError, duration, save_json


def frame_text(image: Image.Image) -> str:
    """Optional bounded OCR of a candidate title card; no service or new AI model."""
    command = shutil.which("tesseract")
    if not command:
        return ""
    buffer = io.BytesIO()
    image = image.copy().convert("RGB")
    image.thumbnail((1280,720))
    image.save(buffer,format="PNG")
    try:
        result = subprocess.run([command,"stdin","stdout","--psm","6"],input=buffer.getvalue(),
                                capture_output=True,check=False,timeout=4)
        return result.stdout.decode("utf-8",errors="ignore")[:4000] if result.returncode == 0 else ""
    except (OSError,subprocess.TimeoutExpired):
        return ""


def matches_title_card(text: str, title: str) -> bool:
    def words(value):
        plain = "".join(char for char in unicodedata.normalize("NFKD",value.casefold()) if not unicodedata.combining(char))
        return re.findall(r"[a-z0-9]+",plain)
    # A supplied release year is metadata, while a numeric title such as 1917
    # or 1923 must remain meaningful title text.
    title = re.sub(r"\s*\((?:19|20)\d{2}\)\s*$","",title)
    expected = [word for word in words(title) if word not in {"the","a","an","and","of","to","in","on","for","with","at","by"}]
    found = set(words(text))
    if not expected:
        return False
    if len(expected) == 1 and len(expected[0]) < 3:
        return any(words(line) == words(title) for line in text.splitlines())
    return sum(word in found for word in expected)/len(expected) >= (1 if len(expected) <= 2 else .8)


def scene_hash(image: Image.Image) -> int | None:
    gray = cv2.cvtColor(np.array(image.resize((160,90))),cv2.COLOR_RGB2GRAY)
    if float(gray.std()) < 8 or float(gray.mean()) < 12:
        return None
    small = cv2.resize(gray,(9,8),interpolation=cv2.INTER_AREA)
    return sum(int(bit)<<index for index,bit in enumerate((small[:,1:]>small[:,:-1]).reshape(-1)))


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
        self.cards = {}
        self.card_texts = {}
        self.titles = {}
        self._scene_hashes = {}
        self._scene_ranges = {}
        self._visual_history = []
        self.people = {}
        self.held = set()
        self._source_intervals = {}
        self._history_clock = 0

    def index(self, path: Path, shots: list[dict]):
        vectors = []
        face_flags = []
        small_frames = []
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
                small_frames.append(image.resize((960,540)))
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
        self._scene_ranges[str(path)] = shots
        self._scene_hashes[str(path)] = [scene_hash(frame) for frame in small_frames]
        self._index_cards(path,shots,small_frames)

    def _index_cards(self, path: Path, shots: list[dict], images: list[Image.Image]):
        """Conservatively flag title/promo cards with the already-loaded CLIP model."""
        try:
            if getattr(self,"_card_text_features",None) is None:
                prompts = ["A movie or television show title card: large title lettering on a plain background.",
                           "A production company, TV channel or streaming service logo on a plain background.",
                           "A cinematic scene with people, places, action, or a dramatic landscape."]
                inputs = self.processor(text=prompts,return_tensors="pt",padding=True,truncation=True)
                with self.torch.inference_mode():
                    vector = self.model.get_text_features(**inputs)
                    if not isinstance(vector,self.torch.Tensor):
                        vector = vector.pooler_output
                    self._card_text_features = vector/vector.norm(dim=-1,keepdim=True)
            with self.torch.inference_mode():
                scores = (self.features[str(path)] @ self._card_text_features.T).numpy()
            self.cards[str(path)] = [
                ("promo" if row[1] > row[0] else "title")
                if max(row[0],row[1]) >= .27 and max(row[0],row[1])-row[2] >= .055 else None
                for row in scores]
        except (AttributeError,RuntimeError,TypeError,ValueError):
            scores = None
            self.cards[str(path)] = [None]*len(self.features[str(path)])
        texts = [""]*len(shots)
        title = getattr(self,"titles",{}).get(str(path),"")
        if title:
            candidates = []
            for index,image in enumerate(images):
                gray = cv2.cvtColor(np.array(image.resize((160,90))),cv2.COLOR_RGB2GRAY)
                plain = float(np.mean(gray < 60)) > .55 and .01 < float(np.mean(gray > 140)) < .4
                score = max(scores[index][0],scores[index][1])-scores[index][2] if scores is not None else -1
                if self.cards[str(path)][index] or plain or score > -.035:
                    candidates.append((score+(.04 if plain else 0),index))
            for _score,index in sorted(candidates,reverse=True)[:12]:
                texts[index] = frame_text(images[index])
                if matches_title_card(texts[index],title):
                    self.cards[str(path)][index] = "title"
        self.card_texts[str(path)] = texts
        for index,kind in enumerate(self.cards[str(path)]):
            if kind:
                self.reserve(path,shots[index]["start"],shots[index]["end"]-shots[index]["start"],"card-exclusion")

    def card_kind(self, path: Path, index: int) -> str | None:
        values = getattr(self,"cards",{}).get(str(path),[])
        return values[index] if index < len(values) else None

    def relevance(self, path: Path, text: str):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77)
        with self.torch.inference_mode():
            vector = self.model.get_text_features(**inputs)
            if not isinstance(vector, self.torch.Tensor):
                vector = vector.pooler_output
            vector = vector / vector.norm(dim=-1, keepdim=True)
            relevance = (self.features[str(path)] @ vector.T).squeeze(-1).numpy()
        return relevance

    def _interval_history(self):
        history = getattr(self, "_source_intervals", None)
        if not isinstance(history, dict):
            history = {}
            self._source_intervals = history
        return history

    def reserve(self, path: Path, start: float, seconds: float, kind: str = "narration"):
        """Share consumed/reserved source windows across previews and every paragraph."""
        if not all(math.isfinite(float(value)) for value in (start, seconds)) or seconds <= 0:
            return
        self._history_clock = int(getattr(self, "_history_clock", 0))+1
        ranges = getattr(self,"_scene_ranges",{}).get(str(path),[])
        hashes = getattr(self,"_scene_hashes",{}).get(str(path),[])
        visual = getattr(self,"_visual_history",[])
        for index,shot in enumerate(ranges):
            if index < len(hashes) and hashes[index] is not None and start < shot["end"] and start+seconds > shot["start"]:
                visual.append((hashes[index],self._history_clock,str(path)))
        self._visual_history = visual
        self._interval_history().setdefault(str(path), []).append(
            {"start": max(0.0, float(start)), "end": max(0.0, float(start))+float(seconds),
             "order": self._history_clock, "kind": kind})

    def unused_windows(self, path: Path, start: float, end: float, gap: float = .18) -> list[tuple[float, float]]:
        """Subtract consumed windows, including a small frame-safety separation."""
        remaining = [(float(start), float(end))] if end > start else []
        for interval in self._interval_history().get(str(path), []):
            beginning, stop = interval["start"]-gap, interval["end"]+gap
            pieces = []
            for low, high in remaining:
                if stop <= low or beginning >= high:
                    pieces.append((low, high))
                else:
                    if beginning > low:
                        pieces.append((low, min(high, beginning)))
                    if stop < high:
                        pieces.append((max(low, stop), high))
            remaining = pieces
        return remaining

    def visual_freshness(self, path: Path, start: float, seconds: float) -> float:
        ranges = getattr(self,"_scene_ranges",{}).get(str(path),[])
        hashes = getattr(self,"_scene_hashes",{}).get(str(path),[])
        relevant = [hashes[i] for i,shot in enumerate(ranges) if i < len(hashes) and hashes[i] is not None
                    and start < shot["end"] and start+seconds > shot["start"]]
        history = getattr(self,"_visual_history",[])
        if not relevant:
            return 1.0
        return sum(not any((value^old).bit_count() <= 6 for old,_order,_path in history) for value in relevant)/len(relevant)

    def freshness(self, path: Path, start: float, seconds: float) -> float:
        if seconds <= 0:
            return 0.0
        return sum(end-beginning for beginning, end in self.unused_windows(path, start, start+seconds, 0))/seconds

    def fresh_seconds(self, path: Path, shots: list[dict]) -> float:
        windows = sorted((float(shot["start"]), float(shot["end"])) for shot in shots if shot["end"] > shot["start"])
        merged = []
        for start, end in windows:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return sum((high-low)*self.visual_freshness(path,low,high-low) for start,end in merged for low,high in self.unused_windows(path,start,end))

    def reuse_penalty(self, path: Path, start: float, seconds: float) -> float:
        """Strong recent-overlap cost, fading only after other source windows play."""
        clock = int(getattr(self, "_history_clock", 0))
        cost = 0.0
        for entry in self._interval_history().get(str(path), []):
            overlap = max(0, min(start+seconds, entry["end"])-max(start,entry["start"]))
            if overlap:
                age = max(0, clock-entry["order"])
                cost += overlap/max(seconds,.1) * (2 if age < 4 else .6/(1+age/8))
        return cost

    def visual_reuse_penalty(self, path: Path, start: float, seconds: float) -> float:
        """Keep a scene's cooldown when the other trailer contains the same image."""
        ranges = getattr(self,"_scene_ranges",{}).get(str(path),[])
        hashes = getattr(self,"_scene_hashes",{}).get(str(path),[])
        relevant = [hashes[i] for i,shot in enumerate(ranges) if i < len(hashes) and hashes[i] is not None
                    and start < shot["end"] and start+seconds > shot["start"]]
        clock = int(getattr(self,"_history_clock",0))
        costs = []
        for value in relevant:
            for old,order,_source in getattr(self,"_visual_history",[]):
                if (value^old).bit_count() <= 6:
                    age = max(0,clock-order)
                    costs.append(2 if age < 4 else .6/(1+age/8))
        return max(costs,default=0.0)

    def hook_score(self, path: Path, shots: list[dict], prompt: str, action: bool) -> float:
        try:
            relevance = self.relevance(path, prompt)
        except (RuntimeError, ValueError, KeyError):
            relevance = np.zeros(len(shots))
        # Optional second sources with unseen footage outrank an exhausted hero shot.
        allowed = [i for i in range(len(shots)) if self.card_kind(path,i) is None] or list(range(len(shots)))
        values = [float(relevance[i]) + (min(shot["motion"],40)/200 if action else 0)
                  - self.reuse_penalty(path,shot["start"],shot["end"]-shot["start"])
                  - self.visual_reuse_penalty(path,shot["start"],shot["end"]-shot["start"])
                  for i,shot in enumerate(shots) if i in allowed]
        return max(values,default=0.0)

    def choose(self, path: Path, shots: list[dict], text: str, seconds: float, action: bool, maximum_shot: float | None = None) -> list[dict]:
        if not shots or seconds <= 0:
            return []
        # Similarity is a soft preference only: weak matches never reject a shot or a video.
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()] or [text]
        weights = np.cumsum([max(1, len(part.split())) for part in sentences])
        remaining = seconds
        chosen = []
        last_index = self.last_shots.get(str(path), -1)
        explicit_limit = maximum_shot is not None
        maximum_shot = maximum_shot or (7.5 if action else 8.5)
        allowed = [i for i in range(len(shots)) if self.card_kind(path,i) is None] or list(range(len(shots)))
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
            candidates = []
            for index, shot in enumerate(shots):
                if index not in allowed:
                    continue
                used = self.uses.get((str(path), index), 0)
                raw_start, raw_end = float(shot["start"]), float(shot["end"])
                fragments = [(low,high) for low,high in self.unused_windows(path,raw_start,raw_end) if high-low >= .75]
                # Exhaustion is allowed. Revisit the least-recently seen interval rather
                # than resetting the semantic hero at every new narration paragraph.
                for start, end in fragments or [(raw_start,raw_end)]:
                    available = max(.1,end-start)
                    unseen = bool(fragments)
                    virgin = self.freshness(path,raw_start,raw_end-raw_start) >= .999
                    visual_fresh = self.visual_freshness(path,start,available) >= .999
                    tier = 2 if virgin and visual_fresh else (1 if unseen and not virgin else 0)
                    score = float(relevance[index]) - used*.04
                    score += min(available,maximum_shot)*.035-min(shot["motion"],60)/600
                    if about_person and str(path) not in held and flags[index]:
                        score += .22
                    score -= self.reuse_penalty(path,start,min(remaining,available,maximum_shot))
                    score -= self.visual_reuse_penalty(path,start,min(remaining,available,maximum_shot))
                    if index == last_index:
                        score -= .35
                    candidates.append((tier,score,-index,index,start,end,not unseen))
            _tier,_score,_order,index,start,end,reused = max(candidates,key=lambda row: row[:3])
            shot = shots[index]
            available = max(.1,end-start)
            portrait = about_person and str(path) not in held and flags[index] and remaining >= 3
            # After genuine exhaustion, a longer calm beat spaces necessary reuse.
            beat_limit = maximum_shot
            if reused and not explicit_limit:
                beat_limit = 10.5 if action else 11.5
            freeze = 0 if portrait else max(0,min(4.5,remaining)-available)
            if reused:
                freeze = max(freeze,min(2.5,max(0,remaining-available)))
            length = min(remaining,3.2 if portrait else available+freeze,beat_limit)
            if 0 < remaining-length < 2.5:
                freeze += remaining-length
                length = remaining
            freeze = min(freeze,max(0,length-.1))
            chosen_start = (start+end)/2 if portrait else start
            chosen.append({"start":chosen_start,"duration":length,"freeze":freeze,"still":bool(portrait),
                           "visual_reason":"portrait-hold" if portrait else ("spaced-source-reuse" if reused else "fresh-script-match")})
            # A portrait hold consumes that visual beat; motion reserves just its
            # live source span, leaving the rest of a long scene available later.
            self.reserve(path,start if portrait else chosen_start,
                         available if portrait else max(.1,min(available,length-freeze)),
                         "portrait" if portrait else "narration")
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
