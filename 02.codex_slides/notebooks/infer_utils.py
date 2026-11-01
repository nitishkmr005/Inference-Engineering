"""Shared helpers for the inference notebooks.

Every notebook starts with `from infer_utils import *`, so each one runs on its own.
Notebooks still define the helpers they teach (measure_stream in 02, bench_openai in 04);
those definitions shadow the copies here and behave the same.
"""
import os, sys, gc, time, json, asyncio, subprocess, urllib.request, platform
import statistics as st

import pandas as pd
import matplotlib.pyplot as plt

try:
    import torch
except ImportError:          # the reading-only and no-GPU sections work without torch
    torch = None

MODEL = os.environ.get("NB_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")   # 1.5B params, 28 layers, GQA with 2 KV heads
MAX_LEN = 4096                                                      # context per request; smaller leaves more room for KV cache
PORT = 8000
BASE = f"http://localhost:{PORT}"
plt.rcParams.update({"figure.figsize": (8, 4), "axes.grid": True, "grid.alpha": .3})


def env_report():
    # One line per component, so a pasted report is enough to debug a setup.
    lines = [f"Python  {platform.python_version()}"]
    if torch is None:
        lines.append("PyTorch not installed")
    else:
        lines.append(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            lines.append(f"GPU     {p.name}, {p.total_memory / 1e9:.0f} GB, sm_{p.major}{p.minor}, {torch.cuda.device_count()} device(s)")
    for mod in ("vllm", "sglang", "tensorrt_llm"):
        try:
            lines.append(f"{mod:8}{__import__(mod).__version__}")
        except Exception:
            pass
    return "\n".join(lines)


def gpu_mem():
    if torch is None or not torch.cuda.is_available():
        return "no GPU"
    free, total = torch.cuda.mem_get_info()
    return f"GPU memory used: {(total - free) / 1e9:.1f} / {total / 1e9:.1f} GB"


def shutdown(llm):
    # Release the GPU memory held by an offline vLLM engine. If memory still isn't
    # freed (it happens in notebooks), restart the kernel.
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def launch(cmd, logfile, health_url, timeout=900, env=None):
    # Start an engine as a subprocess and block until /health answers.
    log = open(logfile, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})})
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"engine exited; tail {logfile}")
        try:
            if urllib.request.urlopen(health_url, timeout=2).status == 200:
                print(f"ready in {time.time() - t0:.0f}s")
                return proc
        except Exception:
            time.sleep(2)
    proc.terminate()
    raise TimeoutError(f"not healthy; see {logfile}")


def stop(proc, timeout=60):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        proc.wait(timeout)
    gc.collect()


def vllm_serve(*extra, model=None, port=PORT, logfile="vllm_server.log", env=None):
    # `vllm serve` with the notebook defaults plus any extra flags.
    cmd = ["vllm", "serve", model or MODEL, "--port", str(port),
           "--max-model-len", str(MAX_LEN), "--gpu-memory-utilization", "0.85", *map(str, extra)]
    return launch(cmd, logfile, f"http://localhost:{port}/health", env=env)


def measure_stream(messages, base=BASE, model=None, max_tokens=200, **kw):
    # TTFT, TPOT and tokens/s for one streamed chat request.
    from openai import OpenAI
    client = OpenAI(base_url=f"{base}/v1", api_key="not-needed")
    t0 = time.perf_counter(); first = last = None; n = 0; text = []
    stream = client.chat.completions.create(model=model or MODEL, messages=messages, max_tokens=max_tokens,
                                            stream=True, stream_options={"include_usage": True}, **kw)
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            now = time.perf_counter(); first = first or now; last = now
            text.append(chunk.choices[0].delta.content)
        if chunk.usage:
            n = chunk.usage.completion_tokens
    tpot = (last - first) / max(n - 1, 1)
    return {"ttft_ms": (first - t0) * 1e3, "tpot_ms": tpot * 1e3, "tok_per_s": 1 / tpot, "tokens": n,
            "e2e_s": last - t0, "text": "".join(text)}


async def bench_openai(base, model, levels=(1, 8, 32, 64), max_tokens=256, prompt_words=200):
    # One async load generator for any OpenAI-compatible engine.
    from openai import AsyncOpenAI
    ac = AsyncOpenAI(base_url=f"{base}/v1", api_key="none")
    filler = " ".join(["data"] * prompt_words)

    async def one(i):
        t0 = time.perf_counter(); first = last = None; n = 0
        stream = await ac.chat.completions.create(model=model, max_tokens=max_tokens, stream=True,
            stream_options={"include_usage": True}, extra_body={"ignore_eos": True},
            messages=[{"role": "user", "content": f"[{i}] {filler}\nWrite a long story about GPU {i}."}])
        async for c in stream:
            if c.choices and c.choices[0].delta.content:
                now = time.perf_counter(); first = first or now; last = now
            if c.usage:
                n = c.usage.completion_tokens
        return first - t0, (last - first) / max(n - 1, 1), n

    rows = []
    for c in levels:
        await asyncio.gather(*[one(i) for i in range(min(c, 4))])          # warm-up
        t0 = time.perf_counter(); res = await asyncio.gather(*[one(i) for i in range(c)])
        wall = time.perf_counter() - t0; ttft, tpot, ns = zip(*res)
        rows.append({"concurrency": c, "ttft_p50_ms": st.median(ttft) * 1e3, "tpot_p50_ms": st.median(tpot) * 1e3,
                     "per_user_tok_s": 1 / st.median(tpot), "system_tok_s": sum(ns) / wall})
    return pd.DataFrame(rows).round(1)
