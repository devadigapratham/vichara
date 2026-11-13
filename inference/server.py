"""Streaming demo API for vichara-reasoner.

    VICHARA_API_KEYS=key1,key2 python inference/server.py --weights checkpoints/grpo2_correct/final.pt

    curl -N -H "X-API-Key: key1" -H "Content-Type: application/json" \
         -d '{"question": "Tom has 3 apples and buys 5 more. How many does he have?"}' localhost:8000/v1/solve
"""

import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.generate import stream_tokens  # noqa: E402
from reasoning.gsm8k_rl import PROMPT  # noqa: E402

STOPS = ["\nQuestion:", "\nYou are an AI", "\nHuman:", "\nUser:", "<|endoftext|>"]
MAX_QUESTION_CHARS = 2000


class StopFilter:
    """Turns the full decoded text so far into safe deltas: never emits a stop string, even half of one."""

    def __init__(self, stops=STOPS):
        self.stops, self.hold = stops, max(map(len, stops)) - 1
        self.sent, self.stopped = 0, False

    def push(self, text: str, final: bool = False) -> str:
        hits = [i for i in (text.find(s) for s in self.stops) if i >= 0]
        if hits:
            self.stopped, end = True, min(hits)
        elif final:
            end = len(text.rstrip("�"))
        else:
            end = len(text) - self.hold
            if "�" in text:
                end = min(end, text.index("�"))
        delta = text[self.sent : end] if end > self.sent else ""
        self.sent = max(self.sent, end)
        return delta


class RateLimiter:
    def __init__(self, per_minute: int, clock=time.monotonic):
        self.rate, self.clock = per_minute / 60.0, clock
        self.capacity = float(per_minute)
        self.buckets = defaultdict(lambda: (self.capacity, self.clock()))

    def allow(self, key: str) -> bool:
        tokens, last = self.buckets[key]
        now = self.clock()
        tokens = min(self.capacity, tokens + (now - last) * self.rate)
        if tokens < 1:
            self.buckets[key] = (tokens, now)
            return False
        self.buckets[key] = (tokens - 1, now)
        return True


class Solve(BaseModel):
    question: str = Field(min_length=1)
    max_new_tokens: int = Field(default=300, ge=1, le=600)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


def create_app(model, encode, decode, eos_id: int, api_keys: set[str], per_minute: int = 20,
               n_valid: int | None = None) -> FastAPI:
    app = FastAPI(title="vichara-reasoner")
    limiter, lock = RateLimiter(per_minute), threading.Lock()
    metrics = {"requests": 0, "rejected_auth": 0, "rejected_rate": 0, "rejected_input": 0,
               "completed": 0, "tokens": 0, "first_token_s": 0.0, "total_s": 0.0}

    def check(key, body):
        metrics["requests"] += 1
        if key not in api_keys:
            metrics["rejected_auth"] += 1
            raise HTTPException(401, "missing or invalid API key")
        if not limiter.allow(key):
            metrics["rejected_rate"] += 1
            raise HTTPException(429, "rate limit exceeded")
        if len(body.question) > MAX_QUESTION_CHARS or "<|" in body.question:
            metrics["rejected_input"] += 1
            raise HTTPException(400, "question too long or contains reserved token syntax")

    def events(body):
        prompt = encode(PROMPT.format(question=body.question))
        budget = min(body.max_new_tokens, model.cfg.max_seq_len - len(prompt))
        stop, ids, t0, first = StopFilter(), [], time.perf_counter(), None
        with lock:
            for token in stream_tokens(model, prompt, budget, eos_id, body.temperature, n_valid=n_valid):
                if token == eos_id:
                    break
                ids.append(token)
                first = first or time.perf_counter() - t0
                delta = stop.push(decode(ids))
                if delta:
                    yield f"data: {json.dumps({'text': delta})}\n\n"
                if stop.stopped:
                    break
        tail = "" if stop.stopped else stop.push(decode(ids), final=True)
        if tail:
            yield f"data: {json.dumps({'text': tail})}\n\n"
        metrics["completed"] += 1
        metrics["tokens"] += len(ids)
        metrics["first_token_s"] += first or 0.0
        metrics["total_s"] += time.perf_counter() - t0
        yield f"data: {json.dumps({'done': True, 'tokens': len(ids), 'stopped': stop.stopped})}\n\n"

    @app.post("/v1/solve")
    def solve(body: Solve, x_api_key: str | None = Header(default=None)):
        check(x_api_key, body)
        return StreamingResponse(events(body), media_type="text/event-stream")

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/metrics")
    def get_metrics(x_api_key: str | None = Header(default=None)):
        if x_api_key not in api_keys:
            raise HTTPException(401, "missing or invalid API key")
        n = max(metrics["completed"], 1)
        return {**metrics, "mean_first_token_s": metrics["first_token_s"] / n,
                "tokens_per_s": metrics["tokens"] / max(metrics["total_s"], 1e-9)}

    return app


def main():
    import uvicorn
    from tokenizers import Tokenizer

    from model.hf_import import load_qwen2
    from training.pretrain import pick_device

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/qwen2.5-0.5b")
    p.add_argument("--weights", default="checkpoints/grpo2_correct/final.pt")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--per-minute", type=int, default=20)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    keys = {k for k in os.environ.get("VICHARA_API_KEYS", "").split(",") if k}
    if not keys:
        sys.exit("set VICHARA_API_KEYS=key1,key2 first")
    device = pick_device(args.device)
    model = load_qwen2(args.model)
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model = model.to(device).eval()
    tok = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    app = create_app(model, lambda s: tok.encode(s).ids, tok.decode, tok.token_to_id("<|endoftext|>"),
                     keys, args.per_minute, n_valid=tok.get_vocab_size())
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
