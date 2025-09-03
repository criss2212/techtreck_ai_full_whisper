# main.py — TechTreck AI server (Whisper + semantic NLP)
import base64
import json
import time
import re
import os
import tempfile
import subprocess
from typing import List, Dict

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faster_whisper import WhisperModel

# --- App & CORS ---
app = FastAPI(title="TechTreck AI Whisper", version="0.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Whisper STT (CPU by default; set device="cuda" if you run on GPU) ---
whisper = WhisperModel("small", device="cpu", compute_type="int8")

# --- Semantic search / NLP utilities ---
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
DIM = 384
index = faiss.IndexFlatIP(DIM)  # cosine via normalized vectors
corpus: List[str] = []
meta: List[Dict] = []

def split_sentences(text: str) -> List[str]:
    sents = nltk.sent_tokenize(text)
    return [re.sub(r"\s+", " ", s).strip() for s in sents if s.strip()]

def embed_texts(texts: List[str]) -> np.ndarray:
    vecs = embedder.encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)

def add_to_index(session_text: str, session_id: str) -> int:
    sents = split_sentences(session_text)
    if not sents:
        return 0
    vecs = embed_texts(sents)
    index.add(vecs)
    corpus.extend(sents)
    meta.extend([{"session_id": session_id, "text": s} for s in sents])
    return len(sents)

# --- REST: ingest / summary / keywords / search ---
class IngestReq(BaseModel):
    text: str
    session_id: str

@app.post("/ingest")
def ingest(req: IngestReq):
    n = add_to_index(req.text, req.session_id)
    return {"added": n}

class SummaryReq(BaseModel):
    text: str
    max_sentences: int = 5

@app.post("/summary")
def summary(req: SummaryReq):
    sents = split_sentences(req.text)
    if not sents:
        return {"summary": ""}
    V = embed_texts(sents)
    centroid = V.mean(axis=0, keepdims=True)
    scores = (V @ centroid.T).ravel()
    idx = np.argsort(-scores)[: req.max_sentences]
    idx = sorted(idx.tolist())
    return {"summary": " ".join(sents[i] for i in idx)}

class KeywordsReq(BaseModel):
    text: str
    top_k: int = 8

@app.post("/keywords")
def keywords(req: KeywordsReq):
    sents = split_sentences(req.text)
    if not sents:
        return {"keywords": []}
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=1000)
    X = vect.fit_transform(sents)
    tfidf = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vect.get_feature_names_out())
    idx = np.argsort(-tfidf)[: req.top_k]
    return {"keywords": terms[idx].tolist()}

class SearchReq(BaseModel):
    query: str
    k: int = 5

@app.post("/search")
def search(req: SearchReq):
    if len(corpus) == 0:
        return {"results": []}
    qv = embed_texts([req.query])
    D, I = index.search(qv.astype("float32"), req.k)
    res = []
    for i, score in zip(I[0].tolist(), D[0].tolist()):
        if 0 <= i < len(corpus):
            res.append(
                {"text": meta[i]["text"], "session_id": meta[i]["session_id"], "score": float(score)}
            )
    return {"results": res}

@app.get("/health")
def health():
    return {"status": "ok", "indexed": len(corpus)}

# --- Audio helpers ---
def s16le_to_float32(pcm_bytes: bytes) -> np.ndarray:
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

def decode_webm_opus_to_float(webm_blob: bytes, sample_rate: int = 16000) -> np.ndarray:
    """
    Decode WebM/Opus to WAV PCM 16k mono using ffmpeg, then load as float32.
    Requires `ffmpeg` installed and available in PATH.
    """
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as fi:
        fi.write(webm_blob)
        fi.flush()
        webm_path = fi.name
    wav_path = webm_path.replace(".webm", ".wav")

    cmd = ["ffmpeg", "-y", "-i", webm_path, "-ac", "1", "-ar", str(sample_rate), "-f", "wav", wav_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    audio, srate = sf.read(wav_path, dtype="float32", always_2d=False)
    try:
        os.remove(webm_path)
        os.remove(wav_path)
    except Exception:
        pass
    return np.asarray(audio, dtype=np.float32)

# --- WebSocket STT: accepts PCM (desktop) and WebM/Opus (web) ---
@app.websocket("/ws")
async def ws_transcribe(ws: WebSocket):
    await ws.accept()

    pcm_buf = np.zeros(0, dtype=np.float32)
    webm_bytes = bytearray()
    last_flush_time = time.time()

    try:
        while True:
            msg = await ws.receive_text()
            evt = json.loads(msg)
            if evt.get("type") != "audio_chunk":
                continue

            fmt = evt.get("format")            # "pcm_s16le" or "webm_opus"
            sr = int(evt.get("sample_rate", 16000))
            raw = base64.b64decode(evt["content"])
            now = time.time()

            if fmt == "pcm_s16le":
                # desktop/mobile: stream raw PCM 16-bit
                pcm_buf = np.concatenate([pcm_buf, s16le_to_float32(raw)])

                # flush every ~1.0s if enough audio
                if (now - last_flush_time) > 1.0 and len(pcm_buf) > sr * 0.5:
                    segments, _ = whisper.transcribe(pcm_buf, language="ro", vad_filter=True)
                    text = "".join(s.text for s in segments).strip()
                    if text:
                        await ws.send_text(json.dumps({"type": "final", "text": text}))
                    pcm_buf = np.zeros(0, dtype=np.float32)
                    last_flush_time = now

            elif fmt == "webm_opus":
                # web: aggregate WebM/Opus chunks, decode via ffmpeg, flush ~1.2s
                webm_bytes.extend(raw)
                if (now - last_flush_time) > 1.2 and len(webm_bytes) > 8000:
                    audio = decode_webm_opus_to_float(bytes(webm_bytes), sample_rate=sr)
                    segments, _ = whisper.transcribe(audio, language="ro", vad_filter=True)
                    text = "".join(s.text for s in segments).strip()
                    if text:
                        await ws.send_text(json.dumps({"type": "final", "text": text}))
                    webm_bytes = bytearray()
                    last_flush_time = now
            else:
                # unknown format: ignore
                pass

    except WebSocketDisconnect:
        # client closed connection
        pass
