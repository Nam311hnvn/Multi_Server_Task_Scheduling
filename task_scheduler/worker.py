"""Educational HTTP worker: synthetic sleeping jobs, one task at a time.

IMPORTANT: this is a networked demonstration, NOT a production service.
Bind to VM private interface; firewall this port to scheduler VM only.
"""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import math
import os
import threading
import time


class Handler(BaseHTTPRequestHandler):
    lock = threading.Lock()  # one execution at a time on this VM

    def log_message(self, fmt, *args):
        print('[worker]', fmt % args, flush=True)

    def reply(self, code, payload):
        data = json.dumps(payload).encode('utf8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authorized(self):
        token = os.getenv('TASK_SCHED_TOKEN', '')
        header = self.headers.get('Authorization', '')
        if not token or not hmac.compare_digest(header, f'Bearer {token}'):
            self.reply(401, {'error': 'Not authorized'})
            return False
        return True

    def do_GET(self):
        if not self.authorized():
            return
        if self.path == '/health':
            return self.reply(200, {'ok': True, 'node': os.getenv('TASK_SCHED_NODE', 'VM')})
        self.reply(404, {'error': 'Not found'})

    def do_POST(self):
        if not self.authorized():
            return
        if self.path != '/execute':
            return self.reply(404, {'error': 'Not found'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 1 <= length <= 4096:
                raise ValueError('Content-Length out of range')
            request = json.loads(self.rfile.read(length))
            task = request['task']
            delay_ms = request['duration_ms']
            if not isinstance(task, str) or len(task) > 100 or not task:
                raise ValueError('Invalid task id')
            if (isinstance(delay_ms, bool) or not isinstance(delay_ms, (int, float))
                    or not math.isfinite(delay_ms) or not 0 <= delay_ms <= 30000):
                raise ValueError('Invalid duration_ms: allowed 0..30000')
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            return self.reply(400, {'error': str(exc)})
        # Per-worker mutex: one job runs at a time even with ThreadingHTTPServer.
        with self.lock:
            start = time.perf_counter()
            time.sleep(delay_ms / 1000.0)  # emulates task duration, does NOT consume CPU
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.reply(200, {'task': task, 'node': os.getenv('TASK_SCHED_NODE', 'VM'),
                             'elapsed_ms': elapsed_ms})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    args = p.parse_args()
    if not os.getenv('TASK_SCHED_TOKEN'):
        p.error('Set TASK_SCHED_TOKEN environment variable before running')
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Worker listening at {args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
