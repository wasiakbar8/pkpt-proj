from flask import Flask, jsonify, request
import threading

app = Flask(__name__)

# Shared cache with run.py
_ARRAY_CACHE_LOCK = threading.Lock()
_ARRAY_CACHE = {}

@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp

@app.route("/api/clear_cache", methods=["POST", "OPTIONS"])
def clear_cache():
    if request.method == "OPTIONS":
        return ("", 204)
    
    with _ARRAY_CACHE_LOCK:
        _ARRAY_CACHE.clear()
    
    return jsonify({"status": "ok", "message": "Cache cleared"})