#!/usr/bin/env python3
"""Local proxy + static server for Prometheus UI.

Serves index.html, proxies /v1/* to Prometheus server (via SSH tunnel), and
exposes /bench/<backend> to read /home/lyy/downloads/bench_dsv4_<backend>.json
over SSH.

Usage:
    python3 server.py --free-token http://127.0.0.1:8787 --ssh-host 219
"""
import argparse
import http.server
import json
import os
import socketserver
import subprocess
import urllib.request
import urllib.error


HERE = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    free_token_url = "http://127.0.0.1:8787"
    ssh_host = "219"
    bench_dir = "/home/lyy/downloads"
    model_bench_prefix = {
        "DeepSeek-V4-Flash-0731": "dsv4",
        "Qwen3.5-122B-A10B-FP8": "qwen35",
        "Qwen3.5-122B-A10B-NVFP4": "qwen35",
    }
    # model_id -> local tunnel port (server.py forwards /v1/* here)
    model_tunnel = {
        "Tensor-0.1-Flash-35B-A3B": 28200,
        "DeepSeek-V4-Flash-0731": 8790,
        "Qwen3.5-122B-A10B-NVFP4": 8791,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=HERE, **kwargs)

    def _no_cache(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")

    def do_GET(self):
        if self.path == "/v1/models":
            return self._models()
        if self.path.startswith("/v1/"):
            return self._proxy()
        if self.path.startswith("/bench/"):
            return self._bench()
        if self.path == "/stats":
            return self._stats()
        return super().do_GET()

    def do_POST(self):
        if self.path == "/bench/clear":
            return self._bench_clear()
        if self.path.startswith("/v1/"):
            return self._proxy(method="POST")
        self.send_error(404, "Not Found")

    def _models(self):
        import urllib.request
        all_models = []
        for model_id, port in self.model_tunnel.items():
            url = f"http://127.0.0.1:{port}/v1/models"
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=10) as r:
                    data = json.loads(r.read().decode())
                    for m in data.get("data", []):
                        if m not in all_models:
                            all_models.append(m)
            except Exception:
                pass
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._no_cache()
        self.end_headers()
        self.wfile.write(json.dumps({"object": "list", "data": all_models}).encode())

    def _stats(self):
        cmd = (
            "echo '===GPU==='; "
            "nvidia-smi --query-gpu=index,name,memory.used,memory.total,"
            "utilization.gpu,utilization.memory,temperature.gpu,power.draw "
            "--format=csv,noheader,nounits 2>/dev/null; "
            "echo '===CPU==='; nproc; cat /proc/loadavg; "
            "top -bn1 | grep -E '^%Cpu' | head -1; "
            "echo '===MEM==='; free -m | head -2; "
            "echo '===PROC==='; "
            "for p in $(pgrep -f 'prometheus.cli serve'); do "
            "if head -c 7 /proc/$p/cmdline 2>/dev/null | grep -q '^python'; then "
            "ps -p $p -o pid,pcpu,pmem,rss,etime --no-headers; break; fi; done"
        )
        try:
            if self.ssh_host in ("localhost", "127.0.0.1"):
                r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=15)
            else:
                r = subprocess.run(
                    ["ssh", "-o", "ConnectTimeout=8", self.ssh_host, cmd],
                    capture_output=True, text=True, timeout=15,
                )
            if r.returncode != 0:
                self.send_response(502)
                self._no_cache()
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(r.stderr.encode())
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self._no_cache()
            self.end_headers()
            self.wfile.write(r.stdout.encode())
        except subprocess.TimeoutExpired:
            self.send_response(504)
            self.end_headers()
            self.wfile.write(b"ssh timeout")
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def _proxy(self, method="GET"):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else None
        # Route by model in body; default to free_token_url (legacy single-server mode)
        target = self.free_token_url
        if body and method == "POST":
            try:
                payload = json.loads(body)
                model_id = payload.get("model", "")
                port = self.model_tunnel.get(model_id)
                if port:
                    target = f"http://127.0.0.1:{port}"
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        url = target + self.path
        req = urllib.request.Request(url, data=body, method=method)
        for k, v in self.headers.items():
            if k.lower() in ("host", "content-length", "connection"):
                continue
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                self.send_response(r.status)
                for k, v in r.headers.items():
                    if k.lower() in ("transfer-encoding", "connection"):
                        continue
                    self.send_header(k, v)
                self._no_cache()
                self.end_headers()
                while True:
                    chunk = r.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(f"proxy error: {e}".encode())

    def _bench(self):
        # /bench/<backend>            (legacy, defaults to DSV4)
        # /bench/<model>/<backend>    (per-model)
        parts = [p for p in self.path.split("/") if p]
        if len(parts) == 2:
            model_id, backend = "DeepSeek-V4-Flash-0731", parts[1]
        elif len(parts) == 3:
            model_id, backend = parts[1], parts[2]
        else:
            self.send_error(400, "bad bench path")
            return
        if not backend or "/" in backend or backend == "clear":
            self.send_error(400, "bad backend")
            return
        prefix = self.model_bench_prefix.get(model_id)
        if not prefix:
            self.send_error(404, f"no bench mapping for model: {model_id}")
            return
        remote = f"{self.bench_dir}/bench_{prefix}_{backend}.json"
        try:
            r = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=15", self.ssh_host, "cat", remote],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                self.send_response(404)
                self._no_cache()
                self.end_headers()
                self.wfile.write(r.stderr.encode())
                return
            data = r.stdout.strip()
            if not data:
                self.send_response(404)
                self._no_cache()
                self.end_headers()
                return
            json.loads(data)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._no_cache()
            self.end_headers()
            self.wfile.write(data.encode())
        except subprocess.TimeoutExpired:
            self.send_response(504)
            self.end_headers()
            self.wfile.write(b"ssh timeout")
        except json.JSONDecodeError:
            self.send_response(404)
            self._no_cache()
            self.end_headers()
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def _bench_clear(self):
        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1420)
    ap.add_argument("--free-token", default="http://127.0.0.1:8787",
                    help="Prometheus API base URL (use http://127.0.0.1:<port> after ssh -L tunnel)")
    ap.add_argument("--ssh-host", default="219")
    ap.add_argument("--bench-dir", default="/home/lyy/downloads")
    args = ap.parse_args()

    Handler.free_token_url = args.free_token.rstrip("/")
    Handler.ssh_host = args.ssh_host
    Handler.bench_dir = args.bench_dir

    print(f"Prometheus UI:  http://localhost:{args.port}")
    print(f"Prometheus API:  {Handler.free_token_url}")
    print(f"Bench ssh:     {Handler.ssh_host}:{Handler.bench_dir}/bench_dsv4_*.json")
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("0.0.0.0", args.port), Handler) as s:
        s.serve_forever()


if __name__ == "__main__":
    main()
