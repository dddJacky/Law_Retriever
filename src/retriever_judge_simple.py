#!/usr/bin/env python3
"""Simple mock judge backend for testing the startup flow."""
import json
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer


class SimpleJudgeHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path == '/judge.html':
            html = '''
<!doctype html>
<html>
<head><title>Judge Batch Processing - Mock</title></head>
<body><h1>法官/律师 - 批量案情处理模式</h1><p>Mock Backend Running</p></body>
</html>
            '''.strip()
            body = html.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(404)

    def do_POST(self):
        if self.path == '/api/batch':
            body = json.dumps({'results': []}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(404)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7861)
    args = parser.parse_args()

    server = HTTPServer(('0.0.0.0', args.port), SimpleJudgeHandler)
    print(f'Mock Judge backend listening on port {args.port}')
    server.serve_forever()
