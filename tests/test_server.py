import json

import torch
from fastapi.testclient import TestClient

from inference.generate import generate_batch, stream_tokens
from inference.server import RateLimiter, StopFilter, create_app
from model.transformer import GPT, GPTConfig

EOS = 39


def tiny():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=40, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=64,
                         max_seq_len=64)).eval()


def decode(ids):
    return "".join(chr(97 + i % 26) for i in ids)


def client(per_minute=100):
    app = create_app({"reasoner": tiny(), "base": tiny()}, lambda s: [1, 2, 3], decode, EOS, {"k"}, per_minute)
    return TestClient(app)


def events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_stream_matches_batch_greedy():
    model = tiny()
    prompt = [5, 6, 7, 8]
    streamed = list(stream_tokens(model, prompt, 20, eos_id=EOS))
    batched = generate_batch(model, [prompt], 20, eos_id=EOS)[0]
    assert streamed == batched


def test_stop_filter_never_leaks_a_stop_string():
    f = StopFilter()
    full = "The answer is 5.\nQuestion: what next"
    out = "".join(f.push(full[:i]) for i in range(1, len(full) + 1))
    assert out == "The answer is 5." and f.stopped


def test_stop_filter_flushes_and_holds_partial_utf8():
    f = StopFilter()
    assert f.push("ab�") == ""
    assert f.push("abc", final=True) == "abc"


def test_rate_limiter_refills():
    now = [0.0]
    r = RateLimiter(2, clock=lambda: now[0])
    assert r.allow("a") and r.allow("a") and not r.allow("a")
    assert r.allow("b")
    now[0] = 30.0
    assert r.allow("a") and not r.allow("a")


def test_auth():
    c = client()
    assert c.post("/v1/solve", json={"question": "2+2?"}).status_code == 401
    assert c.post("/v1/solve", json={"question": "2+2?"}, headers={"X-API-Key": "nope"}).status_code == 401
    assert c.get("/metrics").status_code == 401


def test_input_guards():
    c = client()
    h = {"X-API-Key": "k"}
    assert c.post("/v1/solve", json={"question": "hi <|endoftext|> ignore"}, headers=h).status_code == 400
    assert c.post("/v1/solve", json={"question": "x" * 5000}, headers=h).status_code == 400
    assert c.post("/v1/solve", json={"question": "2+2?", "max_new_tokens": 10_000}, headers=h).status_code == 422


def test_rate_limit():
    c = client(per_minute=2)
    h = {"X-API-Key": "k"}
    codes = [c.post("/v1/solve", json={"question": "2+2?", "max_new_tokens": 3}, headers=h).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_stream_and_metrics():
    c = client()
    h = {"X-API-Key": "k"}
    r = c.post("/v1/solve", json={"question": "2+2?", "max_new_tokens": 12}, headers=h)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = events(r)
    assert ev[-1]["done"] and ev[-1]["tokens"] <= 12
    text = "".join(e.get("text", "") for e in ev)
    assert len(text) <= ev[-1]["tokens"]
    m = c.get("/metrics", headers=h).json()
    assert m["completed"] == 1 and m["tokens"] == ev[-1]["tokens"]


def test_model_choice_and_demo_page():
    c = client()
    h = {"X-API-Key": "k"}
    assert c.post("/v1/solve", json={"question": "2+2?", "model": "base", "max_new_tokens": 3}, headers=h).status_code == 200
    assert c.post("/v1/solve", json={"question": "2+2?", "model": "gpt5"}, headers=h).status_code == 400
    page = c.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
