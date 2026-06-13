import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3')

from transformers import AutoTokenizer
from model.modeling_chatglm import ChatGLMForConditionalGeneration 
from model.configuration_chatglm import ChatGLMConfig
import sys
import pdb
import logging
import math
import json
import torch
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from argparse import ArgumentParser

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    BottleneckConfig,
)
import faiss
import pickle
import argparse
from text2vec import SentenceModel
import errno


def format_law_item(item):
    if isinstance(item, str):
        return item
    if isinstance(item, (list, tuple)):
        return '\n'.join(map(str, item))
    return str(item)


def retrieve_law(query, t2v_model, index, raw_law_data, args_retriver):
    q_emb = t2v_model.encode([query])
    D, I = index.search(q_emb, args_retriver.top_k)
    items = []
    for score, idx in zip(D[0], I[0]):
        item = raw_law_data[idx]
        text = format_law_item(item)
        severity = law_severity_score(text)
        items.append({
            "text": text,
            "score": float(score),
            "severity": severity,
        })
    items.sort(key=lambda x: (-x["severity"], -x["score"]))
    return items


def law_severity_score(text):
    text_lower = text.lower()
    score = 0
    severe_keywords = ['死刑', '无期徒刑', '有期徒刑', '拘役', '刑事', '刑法', '判处', '监禁', '拘留']
    moderate_keywords = ['罚款', '赔偿', '处罚', '责任', '行政处罚', '违约', '民事责任', '刑罚']
    for word in severe_keywords:
        if word in text_lower:
            score += 30
    for word in moderate_keywords:
        if word in text_lower:
            score += 12
    if '合同' in text_lower or '仲裁' in text_lower:
        score -= 4
    if '劳动' in text_lower or '合同' in text_lower:
        score -= 1
    return score


def classify_case(case_text, law_items=None, law_query=''):
    """Very small rule-based classifier to assign a broad case category.
    Returns one of: '刑事', '民事', '行政', '交通', '劳动', '合同', '家庭', '其他'.
    """
    txt = (case_text or '') + ' ' + (law_query or '')
    txt = txt.lower()
    law_blob = ''
    if law_items:
        law_blob = ' '.join((str(x.get('text', '')) for x in law_items))
    txt += ' ' + law_blob.lower()

    criminal = ['故意伤害', '故意杀人', '盗窃', '抢劫', '诈骗', '境外间谍', '刑事', '拘役', '监禁', '死刑', '缓刑', '无期徒刑', '刑法', '判处']
    civil = ['合同', '违约', '赔偿', '民事', '财产损失', '财产', '精神损害', '医患纠纷', '债务', '借款', '侵权', '校园霸凌', '经济纠纷', '产权']
    administrative = ['行政处罚', '罚款', '行政', '行政复议', '行政诉讼', '偷税漏税', '逃税', '税务', '食品安全', '工商', '市场', '卫生', '假账']
    traffic = ['交通', '酒驾', '追尾', '肇事', '超速', '违章', '事故']
    labor = ['劳动', '工资', '劳动仲裁', '工伤', '社保', '劳动合同']
    family = ['离婚', '抚养', '继承', '家庭', '婚姻纠纷', '子女', '父母']

    for kw in criminal:
        if kw in txt:
            return '刑事'
    for kw in traffic:
        if kw in txt:
            return '交通'
    for kw in administrative:
        if kw in txt:
            return '行政'
    for kw in labor:
        if kw in txt:
            return '劳动'
    for kw in family:
        if kw in txt:
            return '家庭'
    for kw in civil:
        if kw in txt:
            return '民事'

    # fallback: if any law item looks severe, tag as 刑事
    if law_items:
        for it in law_items:
            t = str(it.get('text', '')).lower()
            if any(k in t for k in ['死刑', '有期徒刑', '无期徒刑', '监禁']):
                return '刑事'

    return '其他'


def extract_keywords(case_text):
    terms = [
        '故意伤害', '故意杀人', '盗窃', '抢劫', '诈骗', '贩毒', '合同', '违约', '劳动',
        '交通', '酒驾', '行政处罚', '民事', '财产', '继承', '离婚', '抚养', '事故',
        '精神损害', '医患', '债务', '借款', '侵权', '合同诈骗', '刑法', '刑事',
        '民法典', '行政法', '婚姻家庭', '仲裁', '公司法', '知识产权'
    ]
    if not case_text:
        return ''
    text = case_text.lower()
    found = []
    for term in terms:
        if term.lower() in text:
            found.append(term)
    if found:
        return ' / '.join(dict.fromkeys(found))
    # fallback: take first 8 meaningful characters and remove spaces/punctuation
    simple = ''.join(ch for ch in case_text if ch.isalnum() or ch.isspace()).strip()
    parts = simple.split()
    if not parts:
        return case_text[:12]
    if len(parts) == 1:
        return parts[0][:12]
    return ' / '.join(parts[:2])


def get_device_map(num_layers):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available; GPU mode is required for this backend.')

    device_ids = list(range(torch.cuda.device_count()))
    if not device_ids:
        raise RuntimeError('No CUDA devices detected for GPU mode.')

    device_map = {}
    device_map['transformer.word_embeddings'] = device_ids[0]
    device_map['transformer.final_layernorm'] = device_ids[-1]
    device_map['lm_head'] = device_ids[0]

    allocations = [
        device_ids[i] for i in
        sorted(list(range(len(device_ids))) * math.ceil(num_layers / len(device_ids)))
    ]
    allocations = allocations[len(allocations) - num_layers:]
    for layer_i, device_id in enumerate(allocations):
        device_map[f'transformer.layers.{layer_i}.input_layernorm'] = device_id
        device_map[f'transformer.layers.{layer_i}.attention.rotary_emb'] = device_id
        device_map[f'transformer.layers.{layer_i}.attention.query_key_value'] = device_id
        device_map[f'transformer.layers.{layer_i}.attention.dense'] = device_id
        device_map[f'transformer.layers.{layer_i}.post_attention_layernorm'] = device_id
        device_map[f'transformer.layers.{layer_i}.mlp.dense_h_to_4h'] = device_id
        device_map[f'transformer.layers.{layer_i}.mlp.dense_4h_to_h'] = device_id

    return device_map


def main():
    parser = ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--model_path', type=str, default="./model")
    parser.add_argument('--peft_path', type=str, default='./peft_r_model/1.p')
    parser.add_argument('--adapter_path', type=str, default='')
    parser.add_argument('--lora_use', type=bool, default=True)
    parser.add_argument('--adapter_use', type=bool, default=False)
    parser.add_argument('--embedding_path', default='./retriver/law_embs.pkl', type=str, help='')
    parser.add_argument('--rawdata_path', default='./retriver/fatiao.json', type=str, help='核心法条文件')
    parser.add_argument('--top_k', type=int, default=3, help='检索返回条目数量')
    args = parser.parse_args()
    args_retriver = args

    print(f'Using CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")}')

    law_embeds = pickle.load(open(args_retriver.embedding_path, 'rb'))
    raw_law_data = json.load(open(args_retriver.rawdata_path, 'rb'))

    print('load retriver model')
    index = faiss.IndexFlatIP(law_embeds.shape[-1])
    print(index.is_trained)
    index.add(law_embeds)
    print(index.ntotal)

    t2v_model = SentenceModel("./text2vec-base-chinese")

    def read_json(path):
        with open(path, "r") as f:
            return json.load(f)

    logger = logging.getLogger(__file__)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model_class = ChatGLMForConditionalGeneration

    logger.info("Setup Model")
    num_layers = read_json(os.path.join(args.model_path, "config.json"))["num_layers"]
    device_map = get_device_map(num_layers)
    use_cuda = True

    if args.lora_use:
        model_class = ChatGLMForConditionalGeneration
        model = model_class.from_pretrained(args.model_path, device_map=device_map)
        if use_cuda:
            model = model.half()
            model.config.use_cache = True
        else:
            model.config.use_cache = False
        logger.info("Setup PEFT")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=['query_key_value'],
        )
        model = get_peft_model(model, peft_config)

        if use_cuda:
            for layer_i in range(len(model.base_model.model.transformer.layers)):
                device = model.base_model.model.transformer.layers[layer_i].attention.query_key_value.weight.device
                model.base_model.model.transformer.layers[layer_i].attention.query_key_value.lora_B.half().to(device)
                model.base_model.model.transformer.layers[layer_i].attention.query_key_value.lora_A.half().to(device)

        if os.path.exists(args.peft_path):
            model.load_state_dict(torch.load(args.peft_path), strict=False)
    elif args.adapter_use:
        model_class = ChatGLMForConditionalGeneration
        model = model_class.from_pretrained(args.model_path, device_map=device_map)
        if use_cuda:
            model = model.half()
            model.config.use_cache = True
        else:
            model.config.use_cache = False
        logger.info("Setup PEFT")
        peft_config = BottleneckConfig(
            bottleneck_size=512,
            non_linearity='tanh',
            adapter_dropout=0.1,
            use_parallel_adapter=True,
            use_adapterp=False,
            target_modules={"dense_h_to_4h": "mh_adapter", "dense_4h_to_h": "output_adapter"},
            scaling=1.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

        if use_cuda:
            for layer_i in range(len(model.base_model.model.transformer.layers)):
                device = model.base_model.model.transformer.layers[layer_i].mlp.dense_h_to_4h.weight.device
                model.base_model.model.transformer.layers[layer_i].mlp.dense_h_to_4h.adapter_down.half().to(device)
                model.base_model.model.transformer.layers[layer_i].mlp.dense_h_to_4h.adapter_up.half().to(device)
                model.base_model.model.transformer.layers[layer_i].mlp.dense_4h_to_h.adapter_down.half().to(device)
                model.base_model.model.transformer.layers[layer_i].mlp.dense_4h_to_h.adapter_up.half().to(device)

        if os.path.exists(args.adapter_path):
            model.load_state_dict(torch.load(args.adapter_path), strict=False)
    else:
        model_class = ChatGLMForConditionalGeneration
        model = model_class.from_pretrained(args.model_path, device_map=device_map)
        if use_cuda:
            model = model.half()
            model.config.use_cache = True
        else:
            model.config.use_cache = False

    model.eval()

    def handle_query(case_description):
        history = []
        effective_case = case_description
        retrieval_query = case_description
        response, history = model.chat(tokenizer, effective_case + '请给出法律依据', history=history)
        law_items = retrieve_law(retrieval_query, t2v_model, index, raw_law_data, args_retriver)
        law_text = '\n\n'.join(
            f"{idx+1}. [严重度:{item['severity']}] {item['text']}"
            for idx, item in enumerate(law_items)
        )
        prompt = (
            '请根据以下法律条文，生成合理答复。问题是：' + effective_case + '\n' +
            '\n'.join(f"{idx+1}、{item['text']}" for idx, item in enumerate(law_items))
        )
        final_answer, history = model.chat(tokenizer, prompt, history=history)
        analysis_prompt = (
            '请根据上述法律条文，对以下案情进行分析，说明关键法律点：' + effective_case
        )
        analysis, history = model.chat(tokenizer, analysis_prompt, history=history)
        opinion_prompt = (
            '请根据上述分析给出处理意见，语言精炼，便于实际参考。'
        )
        opinion, history = model.chat(tokenizer, opinion_prompt, history=history)
        return final_answer, law_text, analysis, opinion, law_items, extract_keywords(case_description)

    def handle_batch(case_descriptions):
        results = []
        for case_description in case_descriptions:
            case_text = case_description.strip()
            if not case_text:
                results.append({
                    'case_description': case_text,
                    'law_text': '',
                    'analysis': '',
                    'opinion': '',
                    'law_items': [],
                    'keywords': '',
                    'category': '',
                })
                continue
            final_answer, law_text, analysis, opinion, law_items, keywords = handle_query(case_text)
            results.append({
                'case_description': case_text,
                'law_text': law_text,
                'analysis': analysis,
                'opinion': opinion,
                'law_items': law_items,
                'keywords': keywords,
                'category': classify_case(case_text, law_items, ''),
            })
        return results

    class LawRetrievalHandler(BaseHTTPRequestHandler):
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
            # Serve judge page and static files from ./web
            req_path = self.path.split('?', 1)[0]
            web_dir = os.path.join(os.path.dirname(__file__), 'web')
            if req_path in ('/', '/judge.html'):
                try:
                    with open(os.path.join(web_dir, 'judge_index.html'), 'r', encoding='utf-8') as f:
                        content = f.read()
                    self._send_html(content)
                except Exception as e:
                    self.send_error(500, f'File error: {e}')
                return

            # serve JS/CSS/static files
            if req_path.startswith('/'):
                fname = req_path.lstrip('/')
                fs_path = os.path.join(web_dir, fname)
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
            except Exception as exc:
                self.send_error(400, f'Invalid JSON payload: {exc}')
                return
            if not isinstance(case_descriptions, list) or len(case_descriptions) == 0:
                self.send_error(400, 'Empty case_descriptions')
                return
            if len(case_descriptions) < 5:
                case_descriptions = case_descriptions + [''] * (5 - len(case_descriptions))
            case_descriptions = case_descriptions[:5]
            print(f'Received case_descriptions: {case_descriptions}')
            results = handle_batch(case_descriptions)
            self._send_json({
                'case_descriptions': case_descriptions,
                'results': results,
            })

    host = args.host
    port = args.port
    try:
        server = HTTPServer((host, port), LawRetrievalHandler)
    except OSError as e:
        if getattr(e, 'errno', None) == errno.EADDRINUSE:
            print(f'端口 {port} 已被占用，请先关闭占用进程后重试。')
            return
        raise

    listen_url = f'http://127.0.0.1:{port}/judge.html'
    print(f'法官/律师批量处理页面已启动，请访问：{listen_url}')
    try:
        webbrowser.open(listen_url)
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n服务器已停止。')
    finally:
        try:
            server.server_close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
