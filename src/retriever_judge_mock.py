#!/usr/bin/env python3
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

WEB_DIR = os.path.join(os.path.dirname(__file__), 'web')

class MockHandler(BaseHTTPRequestHandler):
    def _send_html(self, html):
        body = html.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return

    def do_GET(self):
        req = self.path.split('?', 1)[0]
        if req in ('/', '/judge.html'):
            try:
                with open(os.path.join(WEB_DIR, 'judge_index.html'), 'r', encoding='utf-8') as f:
                    self._send_html(f.read())
            except Exception as e:
                self.send_error(500, f'File error: {e}')
            return

        if req.startswith('/'):
            fname = req.lstrip('/')
            fs_path = os.path.join(WEB_DIR, fname)
            if os.path.exists(fs_path) and os.path.isfile(fs_path):
                try:
                    with open(fs_path, 'rb') as f:
                        data = f.read()
                    if fname.endswith('.js'):
                        ctype = 'application/javascript'
                    elif fname.endswith('.css'):
                        ctype = 'text/css'
                    else:
                        ctype = 'application/octet-stream'
                    self.send_response(200)
                    self.send_header('Content-Type', f'{ctype}; charset=utf-8')
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except Exception as e:
                    self.send_error(500, f'File read error: {e}')
                return

        self.send_error(404, 'Not Found')

    def do_POST(self):
        if self.path != '/api/batch':
            self.send_error(404, 'Not Found')
            return
        length = int(self.headers.get('Content-Length', 0))
        payload = self.rfile.read(length)
        try:
            request_data = json.loads(payload.decode('utf-8'))
            case_descriptions = request_data.get('case_descriptions', [])
            law_query = request_data.get('law_query', '')
        except Exception as exc:
            self.send_error(400, f'Invalid JSON payload: {exc}')
            return

        if not isinstance(case_descriptions, list):
            self.send_error(400, 'case_descriptions must be a list')
            return

        # Build mock results
        results = []
        for i in range(5):
            desc = ''
            if i < len(case_descriptions):
                desc = case_descriptions[i]
            category = '其他'
            if desc:
                if any(k in desc for k in ['故意', '杀', '伤', '骗', '盗']):
                    category = '刑事'
                elif any(k in desc for k in ['合同', '违约', '赔偿']):
                    category = '民事'
            results.append({
                'case_description': desc,
                'law_items': [
                    {'text': f'示例法条 {i+1}：关于...，与关键词 {law_query} 相关。', 'severity': 10}
                ] if desc else [],
                'analysis': f'（模拟）对案情 {i+1} 的简要分析。',
                'opinion': f'（模拟）对案情 {i+1} 的处理意见。',
                'keywords': law_query,
                'law_query': law_query,
                'category': category,
            })

        self._send_json({'results': results, 'law_query': law_query})


def run():
    host = '0.0.0.0'
    port = 7861
    server = HTTPServer((host, port), MockHandler)
    listen_url = f'http://127.0.0.1:{port}/judge.html'
    print(f'Mock server running at: {listen_url}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == '__main__':
    run()
