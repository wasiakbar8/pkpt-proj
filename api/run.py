from http.server import BaseHTTPRequestHandler
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import median
from typing import Any, Dict, Tuple
import numpy as np

# Cache
_ARRAY_CACHE_LOCK = threading.Lock()
_ARRAY_CACHE: Dict[int, Tuple] = {}


def _get_arrays(n: int):
    with _ARRAY_CACHE_LOCK:
        if n in _ARRAY_CACHE:
            return _ARRAY_CACHE[n]
        rng = np.random.default_rng(42)
        A = np.ascontiguousarray(rng.standard_normal((n, n), dtype=np.float32))
        B = np.ascontiguousarray(rng.standard_normal((n, n), dtype=np.float32))
        OUT = np.ascontiguousarray(np.empty((n, n), dtype=np.float32))
        _ARRAY_CACHE[n] = (A, B, OUT)
        return A, B, OUT


def _kernel_chunk(A, B, OUT, i0: int, i1: int):
    a = A[i0:i1]
    b = B[i0:i1]
    np.multiply(a, b, out=OUT[i0:i1])
    OUT[i0:i1] += np.sin(a)
    OUT[i0:i1] -= np.sqrt(np.abs(b))


def _run_once(threads: int, chunk_size: int, matrix_size: int) -> float:
    A, B, OUT = _get_arrays(matrix_size)
    n = matrix_size
    chunk = max(1, min(chunk_size, n))
    ranges = [(i, min(i + chunk, n)) for i in range(0, n, chunk)]

    t0 = time.perf_counter()

    if threads <= 1:
        for i0, i1 in ranges:
            _kernel_chunk(A, B, OUT, i0, i1)
    else:
        workers = min(threads, len(ranges))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_kernel_chunk, A, B, OUT, i0, i1) for (i0, i1) in ranges]
            for f in as_completed(futs):
                f.result()

    return time.perf_counter() - t0


def _benchmark(threads: int, chunk_size: int, matrix_size: int, repeats: int) -> Dict[str, Any]:
    # warm-up
    _run_once(threads, chunk_size, matrix_size)
    times = [_run_once(threads, chunk_size, matrix_size) for _ in range(repeats)]
    return {
        "times": [float(x) for x in times],
        "median": float(median(times)),
        "min": float(min(times)),
        "max": float(max(times)),
        "repeats": repeats,
    }


def _clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(x)
    except Exception:
        return default
    return max(lo, min(hi, v))


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/api/debug':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b'I am working!')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != '/api/run':
            self.send_response(404)
            self.end_headers()
            return

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body) if body else {}

            threads = _clamp_int(data.get("thread_count"), 1, 64, 4)
            chunk_size = _clamp_int(data.get("chunk_size"), 1, 4096, 128)
            matrix_size = _clamp_int(data.get("matrix_size"), 128, 4096, 1024)
            repeats = _clamp_int(data.get("repeats"), 1, 15, 5)

            base = _benchmark(1, chunk_size, matrix_size, repeats)
            cur = _benchmark(threads, chunk_size, matrix_size, repeats)

            t1 = float(base["median"])
            tp = float(cur["median"])
            speedup = (t1 / tp) if tp > 0 else 0.0
            efficiency = (speedup / threads) if threads > 0 else 0.0

            result = {
                "config": {
                    "thread_count": threads,
                    "chunk_size": chunk_size,
                    "matrix_size": matrix_size,
                    "repeats": repeats,
                },
                "baseline": base,
                "current": cur,
                "metrics": {
                    "baseline_median_s": t1,
                    "current_median_s": tp,
                    "speedup": speedup,
                    "efficiency": efficiency,
                },
            }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            error_response = {"error": str(e)}
            self.wfile.write(json.dumps(error_response).encode())