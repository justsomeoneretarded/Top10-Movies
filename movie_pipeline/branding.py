"""Original Movie Ranks ident: folded M, restrained light, original two-hit audio."""
from __future__ import annotations
from pathlib import Path
import wave
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

IDENT_SECONDS = 2.3

def logo_image(path: Path, width: int = 1280, height: int = 720):
    from .render import brand_font
    canvas = Image.new("RGB", (width, height), "#09090c")
    unit = height/720
    left, top = round(width/2-158*unit), round(136*unit)
    mask = Image.new("L", canvas.size)
    d = ImageDraw.Draw(mask)
    shapes = [[(0,0),(76,0),(76,278),(0,294)],
              [(76,0),(158,137),(158,234),(0,0)],
              [(158,137),(240,0),(316,0),(158,234)],
              [(240,0),(316,0),(316,294),(240,278)]]
    colors = [(135,4,19),(239,18,39),(190,8,28),(125,2,18)]
    for poly, color in zip(shapes, colors):
        pts = [(left+round(x*unit), top+round(y*unit)) for x,y in poly]
        layer = Image.new("L", canvas.size)
        ImageDraw.Draw(layer).polygon(pts, fill=255)
        y = np.linspace(0,1,height)[:,None,None]
        rgb = np.clip(np.asarray(color)[None,None,:]*(1.25-.5*y),0,255)
        gradient = Image.fromarray(np.repeat(rgb,width,axis=1).astype("uint8"))
        canvas.paste(gradient, (0,0), layer)
        ImageDraw.Draw(canvas).line([*pts,pts[0]], fill="#f5505b", width=max(1,round(unit)))
    d = ImageDraw.Draw(canvas)
    label = "MOVIE RANKS"
    font = brand_font("body-medium", round(30*unit))
    d.text((width/2, top+342*unit), label, font=font, anchor="mm", fill="#e9e9ed", stroke_width=0)
    canvas.save(path)
    return canvas

def render_ident(output: Path, work: Path, seconds: float = IDENT_SECONDS):
    from .core import ffmpeg
    from .render import LOSSLESS_AUDIO
    from .video_settings import export_size
    work.mkdir(parents=True, exist_ok=True)
    width,height = export_size()
    picture = work/"movie-ranks-ident.png"
    logo_image(picture, width, height)
    rate=48000
    t=np.arange(round(rate*seconds))/rate
    audio=np.zeros(len(t), dtype=np.float64)
    # Two brief low-level cinematic notes; no noise bed and no borrowed recording.
    for at,frequency,level in ((.30,146.83,.13),(1.04,220.0,.09)):
        x=np.maximum(0,t-at)
        envelope=np.minimum(1,x/.015)*np.exp(-x*10)*(t>=at)*(x<.65)
        audio += level*envelope*(np.sin(2*np.pi*frequency*x)+.2*np.sin(2*np.pi*frequency*2*x))
    wav=work/"ident.wav"
    with wave.open(str(wav),"wb") as stream:
        stream.setnchannels(2);stream.setsampwidth(2);stream.setframerate(rate)
        stream.writeframes((np.repeat(audio[:,None],2,axis=1)*32767).astype("<i2").tobytes())
    ffmpeg("-loop","1","-framerate","30","-i",picture,"-i",wav,
           "-vf",f"fade=t=in:d=0.28,fade=t=out:st={seconds-.3}:d=0.3",
           "-t",seconds,"-r","30","-c:v","libx264","-preset","veryfast","-crf","18",
           "-pix_fmt","yuv420p","-threads","2",*LOSSLESS_AUDIO,output)
    return seconds
