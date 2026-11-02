"""Bounded web-image discovery with verified TMDB stills as an independent-photo fallback.

No extracted video frame is ever returned by this module. Search failure is optional.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse, urljoin
import hashlib
import html
import ipaddress
import json
import re
import requests
from PIL import Image
from .editorial_policy import image_queries
from .core import normalized, save_json

MAX_BYTES = 10_000_000
BAD_CONTEXT = re.compile(r"\b(ai generated|concept art|fan art|wallpaper ai|reaction|review thumbnail|youtube thumbnail)\b", re.I)

def public_url(url: str) -> bool:
    p=urlparse(url)
    if p.scheme not in ("http","https") or not p.hostname or p.username or p.password:
        return False
    host=p.hostname.lower()
    if host in {"localhost","metadata.google.internal"} or host.endswith((".local",".internal")):
        return False
    try:
        return ipaddress.ip_address(host).is_global
    except ValueError:
        return "." in host

def search(film: dict) -> list[dict]:
    rows=[]
    for query in image_queries(film, limit=2):
        try:
            response=requests.get("https://www.bing.com/images/search", params={"q":query},
                                  headers={"User-Agent":"Mozilla/5.0"}, timeout=(4,8))
            response.raise_for_status()
            for raw in re.findall(r'\bm="([^"]+)"',response.text):
                try:
                    item=json.loads(html.unescape(raw))
                except (ValueError,TypeError):
                    continue
                url,page=item.get("murl",""),item.get("purl","")
                context=" ".join(str(item.get(k,"")) for k in ("t","desc","purl"))
                if not public_url(url) or BAD_CONTEXT.search(context):
                    continue
                if any(host in url+" "+page for host in ("ytimg.com","youtube.com","pinterest.com")):
                    continue
                words=set(normalized(film["title"]).split())
                found=set(normalized(context).split())
                if len(words & found)/max(1,len(words)) < .7:
                    continue
                # Movie remakes require corroborating year context. Shows require title context;
                # requested season, when present, is part of the search query.
                if film.get("kind") != "tv" and str(film.get("year", "")) not in context:
                    continue
                rows.append({"url":url,"page":page,"query":query,"context":context[:300]})
                if len(rows)>=4:
                    return rows
        except requests.RequestException:
            continue
    return rows

def read_image(url: str, cache: Path):
    if not public_url(url):
        return None
    target=cache/(hashlib.sha256(url.encode()).hexdigest()[:24]+".jpg")
    if target.exists():
        try:
            return Image.open(target).convert("RGB")
        except OSError:
            return None
    cache.mkdir(parents=True,exist_ok=True)
    try:
        for _ in range(4):
            if not public_url(url):
                return None
            response=requests.get(url,timeout=(4,8),stream=True,allow_redirects=False,
                                  headers={"User-Agent":"Mozilla/5.0"})
            if response.status_code in (301,302,303,307,308):
                url=urljoin(url,response.headers.get("Location",""))
                response.close()
                continue
            response.raise_for_status()
            if not response.headers.get("Content-Type","").startswith("image/"):
                response.close();return None
            data=bytearray()
            for block in response.iter_content(65536):
                data.extend(block)
                if len(data)>MAX_BYTES:
                    response.close();return None
            response.close()
            image=Image.open(BytesIO(data))
            if image.width*image.height>40_000_000 or image.width<800 or image.height<450:
                return None
            image=image.convert("RGB")
            image.thumbnail((1800,1800),Image.Resampling.LANCZOS)
            image.save(target,quality=94)
            return image
    except (requests.RequestException,OSError,ValueError,Image.DecompressionBombError):
        return None
    return None

def gather_photos(films: list[dict], work: Path) -> tuple[list, list[dict]]:
    from .thumbnail_studio import Candidate, face_quality, frame_sharpness, has_burned_text, _dhash
    from .stills_trailer import fetch_original
    provenance=[]
    def one(pair):
        index,film=pair
        accepted=[]
        try:
            rows=search(film)
        except (OSError,ValueError,KeyError,requests.RequestException):
            rows=[]
        for row in rows:
            image=read_image(row["url"],work/"web")
            if image is None or frame_sharpness(image)<20 or has_burned_text(image):
                continue
            face,quality,good=face_quality(image)
            candidate=Candidate(image,index,"web-image")
            candidate.face=face
            candidate.score=quality
            candidate.notes={"good_face":good,"face_quality":quality,"hash":_dhash(image),
                             "url":row["url"],"source_page":row["page"],"query":row["query"],
                             "identity_evidence":row["context"],"source_kind":"web-image"}
            accepted.append(candidate)
            if len(accepted)>=2:
                break
        if not accepted:
            for path in film.get("backdrops",[])[:3]:
                try:
                    image=fetch_original(path,work/"publicity")
                    if image is None or image.width<800:
                        continue
                    image=image.convert("RGB");image.thumbnail((1800,1800),Image.Resampling.LANCZOS)
                    face,quality,good=face_quality(image)
                    candidate=Candidate(image,index,"publicity-fallback")
                    candidate.face=face;candidate.score=quality
                    candidate.notes={"good_face":good,"face_quality":quality,"hash":_dhash(image),
                                     "source_kind":"publicity-fallback","url":"https://image.tmdb.org/t/p/original"+path,
                                     "fallback_reason":"No usable independently searched photo",
                                     "identity_evidence":"TMDB image attached to this exact title ID"}
                    accepted.append(candidate)
                    break
                except (OSError,ValueError,requests.RequestException):
                    continue
        return accepted
    with ThreadPoolExecutor(max_workers=3) as pool:
        grouped=list(pool.map(one,enumerate(films)))
    result=[]
    hashes=[]
    for group in grouped:
        for candidate in group:
            value=candidate.notes["hash"]
            if any((value^old).bit_count()<=5 for old in hashes):
                continue
            hashes.append(value);result.append(candidate)
            provenance.append({"film":films[candidate.film]["title"],**candidate.notes})
    save_json(work/"image-discovery.json",provenance)
    return result,provenance
