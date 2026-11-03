"""Matched real-narrator QA; no soundtrack or trailer layer hides the speech."""
from __future__ import annotations
import json
from pathlib import Path
from movie_pipeline.core import duration, voice_reference, ffmpeg
from movie_pipeline.voice import Narrator, paced_take
from movie_pipeline.sound_design import delivery_speed
from movie_pipeline.speech_recovery import SpeechRecovery
from faster_whisper import WhisperModel

TEXT = ("Three friends wake up in Las Vegas with no memory of the night before. "
        "The groom is missing, and they have to find him before the wedding. "
        "They start retracing their steps, but every clue raises another question about what happened. "
        "One simple job turns into an increasingly ridiculous mess. "
        "The story keeps moving because none of them knows which discovery will make their situation worse.")


def main():
    root=Path("natural-voice-review")
    root.mkdir(parents=True,exist_ok=True)
    reference=voice_reference("warguy",Path("assets/voices"))
    recognizer=WhisperModel("base.en",device="cpu",compute_type="int8",cpu_threads=2)
    recovery=SpeechRecovery(recognizer,lambda:Narrator(reference,root/"speech"),root/"speech-checks")
    try:
        take,cues=recovery.recover(TEXT,label="natural narration sample")
    finally:
        recovery.close()
    seconds=duration(take)
    factor=delivery_speed(1.08,"normal",words=len(TEXT.split()),seconds=seconds)
    paced=paced_take(take,factor)
    ffmpeg("-i",take,"-c:a","pcm_s24le",root/"processed.wav")
    dry=take.with_suffix(".dry.wav")
    if dry.exists():
        ffmpeg("-i",dry,"-c:a","pcm_s24le",root/"dry.wav")
    ffmpeg("-i",paced,"-c:a","pcm_s24le",root/"paced.wav")
    report={"text":TEXT,"narrator":"warguy","model":"turbo","words":len(TEXT.split()),
            "recorded_seconds":seconds,"recorded_wpm":len(TEXT.split())*60/seconds,
            "pitch_preserving_factor":factor,"paced_seconds":duration(paced),"captions":cues}
    (root/"voice-review.json").write_text(json.dumps(report,indent=2))
    print("VOICE_SAMPLE",json.dumps(report),flush=True)


if __name__=="__main__":
    main()
