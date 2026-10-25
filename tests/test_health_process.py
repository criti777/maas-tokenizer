"""Exercise both failure paths against a real HTTP server in an isolated process."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="requires POSIX signals")
def test_http_health_reports_stalled_worker_and_times_out_for_stopped_process(tmp_path):
    script = """
import json, socket, threading
import uvicorn
import maas_tokenizer.api as api
release = threading.Event()
def count(request):
    if request.get('block'):
        release.wait(30)
    return 1
api._service.count = count
sock = socket.socket()
sock.bind(('127.0.0.1', 0))
sock.listen(128)
print(json.dumps({'port': sock.getsockname()[1]}), flush=True)
uvicorn.Server(uvicorn.Config(api.app, log_level='warning', access_log=False)).run(sockets=[sock])
"""
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(str(Path(p).resolve()) for p in sys.path),
        "TOKENIZER_MAX_EXECUTION_SECONDS": "0.05",
        "TOKENIZER_LOG_PATH": str(tmp_path / "access.log"),
        "TOKENIZER_RUN_LOG_PATH": str(tmp_path / "run.log"),
    }
    with (tmp_path / "server.log").open("w") as stderr:
        process = subprocess.Popen(
            [sys.executable, "-c", script], env=env,
            stdout=subprocess.PIPE, stderr=stderr, text=True,
        )
        try:
            # Bound startup reads as well as HTTP calls, so a failed child cannot hang pytest.
            with ThreadPoolExecutor(max_workers=1) as pool:
                line = pool.submit(process.stdout.readline)
                try:
                    metadata = json.loads(line.result(timeout=15))
                except BaseException:
                    process.kill()
                    raise
            url = f"http://127.0.0.1:{metadata['port']}/health"
            deadline = time.monotonic() + 15
            while True:
                try:
                    with urllib.request.urlopen(url, timeout=0.5) as response:
                        assert response.status == 200
                    break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() > deadline:
                        raise AssertionError("server did not become healthy")
                    time.sleep(0.05)

            request = urllib.request.Request(
                url.replace('/health', '/tokenizer'),
                data=json.dumps({'model': 'glm-5.2', 'block': True}).encode(),
                headers={'Content-Type': 'application/json'},
            )
            with pytest.raises((TimeoutError, urllib.error.URLError)):
                urllib.request.urlopen(request, timeout=0.3)
            # Client timeout must not clear the still-running computation's health state.
            with pytest.raises(urllib.error.HTTPError) as failure:
                urllib.request.urlopen(url, timeout=2)
            assert failure.value.code == 504
            assert json.load(failure.value)['reason'] == 'execution_timeout'
            failure.value.close()

            os.kill(process.pid, signal.SIGSTOP)
            deadline = time.monotonic() + 2
            while True:
                pid, status = os.waitpid(process.pid, os.WNOHANG | os.WUNTRACED)
                if pid and os.WIFSTOPPED(status):
                    break
                assert time.monotonic() < deadline, "child did not stop"
                time.sleep(0.005)
            # TCP can connect via the kernel while the HTTP process cannot respond.
            with socket.create_connection(('127.0.0.1', metadata['port']), timeout=1):
                pass
            with pytest.raises((TimeoutError, urllib.error.URLError)):
                urllib.request.urlopen(url, timeout=0.3)

            os.kill(process.pid, signal.SIGCONT)
            with pytest.raises(urllib.error.HTTPError) as resumed:
                urllib.request.urlopen(url, timeout=2)
            assert resumed.value.code == 504
            resumed.value.close()
        finally:
            if process.poll() is None:
                # The injected computation is deliberately blocked: never wait for graceful shutdown.
                process.kill()
            process.wait(timeout=5)
            process.stdout.close()
