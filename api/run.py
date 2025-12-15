from __future__ import annotations

import time
import threading
import os
from dataclasses import dataclass
from statistics import median
from typing import Dict, Tuple, Any

import numpy as np
from flask import Flask, jsonify, request, make_response, send_file
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)

# ---- CORS + Preflight ----
@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp

@dataclass(frozen=True)
class BenchConfig:
    thread_count: int
    chunk_size: int
    matrix_size: int
    repeats: int

_ARRAY_CACHE_LOCK = threading.Lock()
_ARRAY_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

def _get_arrays(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with _ARRAY_CACHE_LOCK:
        if n in _ARRAY_CACHE:
            return _ARRAY_CACHE[n]
        rng = np.random.default_rng(42)
        A = np.ascontiguousarray(rng.standard_normal((n, n), dtype=np.float32))
        B = np.ascontiguousarray(rng.standard_normal((n, n), dtype=np.float32))
        OUT = np.ascontiguousarray(np.empty((n, n), dtype=np.float32))
        _ARRAY_CACHE[n] = (A, B, OUT)
        return A, B, OUT

def _kernel_chunk(A: np.ndarray, B: np.ndarray, OUT: np.ndarray, i0: int, i1: int) -> None:
    a = A[i0:i1]
    b = B[i0:i1]
    np.multiply(a, b, out=OUT[i0:i1])
    OUT[i0:i1] += np.sin(a)
    OUT[i0:i1] -= np.sqrt(np.abs(b))

def _run_once(cfg: BenchConfig) -> float:
    A, B, OUT = _get_arrays(cfg.matrix_size)
    n = cfg.matrix_size
    chunk = max(1, min(cfg.chunk_size, n))
    ranges = [(i, min(i + chunk, n)) for i in range(0, n, chunk)]
    t0 = time.perf_counter()
    if cfg.thread_count <= 1:
        for i0, i1 in ranges:
            _kernel_chunk(A, B, OUT, i0, i1)
    else:
        workers = min(cfg.thread_count, len(ranges))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_kernel_chunk, A, B, OUT, i0, i1) for (i0, i1) in ranges]
            for f in as_completed(futs):
                f.result()
    return time.perf_counter() - t0

def _benchmark(cfg: BenchConfig) -> Dict[str, Any]:
    _run_once(BenchConfig(cfg.thread_count, cfg.chunk_size, cfg.matrix_size, 1))
    times = [_run_once(cfg) for _ in range(cfg.repeats)]
    return {
        "times": [float(x) for x in times],
        "median": float(median(times)),
        "min": float(min(times)),
        "max": float(max(times)),
        "repeats": cfg.repeats,
    }

def _clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(x)
    except Exception:
        return default
    return max(lo, min(hi, v))

# --- UPDATED ROUTE ---
# Added "GET" to methods to allow browser loading
@app.route("/", methods=["GET", "POST", "OPTIONS"])
def run_benchmark():
    # 1. Handle OPTIONS (Preflight)
    if request.method == "OPTIONS":
        return make_response("", 204)

    # 2. Handle GET (Serve the HTML file)
    if request.method == "GET":
        # Look for index.html in the parent directory (root)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        html_file = os.path.join(base_dir, 'index.html')
        try:
            return send_file(html_file)
        except FileNotFoundError:
            return "index.html not found. Please check your file structure.", 404

    # 3. Handle POST (Run the Benchmark)
    data = request.get_json(silent=True) or {}
    threads = _clamp_int(data.get("thread_count"), 1, 64, 4)
    chunk_size = _clamp_int(data.get("chunk_size"), 1, 4096, 128)
    matrix_size = _clamp_int(data.get("matrix_size"), 128, 4096, 1024)
    repeats = _clamp_int(data.get("repeats"), 1, 15, 5)

    cfg = BenchConfig(threads, chunk_size, matrix_size, repeats)
    base_cfg = BenchConfig(1, chunk_size, matrix_size, repeats)

    base = _benchmark(base_cfg)
    cur = _benchmark(cfg)

    t1 = float(base["median"])
    tp = float(cur["median"])
    speedup = (t1 / tp) if tp > 0 else 0.0
    efficiency = (speedup / threads) if threads > 0 else 0.0

    return jsonify({
        "config": {
            "thread_count": threads,
            "chunk_size": chunk_size,
            "matrix_size": matrix_size,
            "repeats": repeats
        },
        "baseline": base,
        "current": cur,
        "metrics": {
            "baseline_median_s": t1,
            "current_median_s": tp,
            "speedup": speedup,
            "efficiency": efficiency
        }
    })