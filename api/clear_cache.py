from flask import Flask, jsonify, request, make_response

app = Flask(__name__)

@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return resp

@app.route("/", methods=["POST", "OPTIONS"])
def clear_cache():
    if request.method == "OPTIONS":
        return make_response("", 204)
    # On Vercel serverless, memory isn't guaranteed across calls,
    # so "cache clear" is mostly symbolic—but keep it for UI.
    return jsonify({"ok": True})
