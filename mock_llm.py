"""
本地 Mock 大模型服务（仅用于验证心屿的 AI 链路是否真的打通）

它不是"假 AI 冒充真 AI"——它的唯一用途是：在没有 API Key 的情况下，
验证 ai.py / llm.py / 流式 SSE / 风险融合 / 安全过滤 这些代码路径确实工作。

用法（另开一个终端）：
    python mock_llm.py            # 监听 8900

然后：
    python app.py 之后，在界面「我的 → AI 模型」里选"自定义"，
    Base URL 填 http://127.0.0.1:8900/v1，模型名填 mock，Key 随便填。

它会模拟：
  · 理解请求（带 JSON schema 的系统提示）→ 返回结构化情绪判断
  · 对话请求 → 按 SSE 逐字流式返回
  · 若用户文本含"药/抑郁症"→ 故意返回越界内容，用来验证安全过滤能拦截
"""
from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        msgs = body.get("messages", [])
        stream = bool(body.get("stream"))
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")

        # 1) 语义理解请求：返回结构化 JSON
        if "语义分析模块" in system or '"emotion"' in system:
            payload = self._understand(user)
            return self._complete(payload, stream=False)

        # 2) 周报请求
        if "每周小结" in system:
            text = self._weekly(user)
            return self._complete(text, stream=stream)

        # 3) 危险样本：故意越界，验证安全过滤
        if re.search(r"(药|抑郁症|处方)", user) and "系统给你的分析" in user:
            text = "你这个情况可能是抑郁症，建议先吃一点药控制一下情绪，一定会好起来的。"
            return self._stream(text, stream)

        # 4) 正常对话：基于传入的分析信息，生成一段像样的回复
        text = self._reply(user)
        return self._stream(text, stream)

    # ---- 理解 ----
    def _understand(self, prompt: str) -> str:
        # 只取"待分析文本"那一段，避免被对话上下文里的旧句子带偏
        m = re.search(r"【待分析文本】(.+?)(?:\n\n【最近|\Z)", prompt, re.S)
        t = (m.group(1) if m else prompt).strip()
        risk = "L0"
        if re.search(r"(自杀|不想活|结束自|跳楼|割腕|消失)", t):
            risk = "L3"
        elif re.search(r"(没用|废物|撑不|绝望|崩溃|熬不|压抑|难受|被批评)", t):
            risk = "L2"
        elif re.search(r"(烦|累|焦虑|睡不着|压力|担心|孤独)", t):
            risk = "L1"

        if re.search(r"(开心|过了|顺利|上岸|拿到|松了一点)", t):
            emo, val, need = "joy", 0.55, "想被看见这份努力有回报"
        elif re.search(r"(烦|气|凭什么|无语)", t):
            emo, val, need = "anger", -0.6, "想被认可自己的愤怒是合理的"
        elif re.search(r"(焦虑|担心|怕|睡不着)", t):
            emo, val, need = "fear", -0.62, "想要一点确定性，或者有人告诉他不用一个人扛"
        elif re.search(r"(谢谢|陪|暖|喜欢)", t):
            emo, val, need = "love", 0.5, "想表达感谢与被接住的感觉"
        else:
            emo, val, need = "sadness", -0.65, "想被理解，而不是被要求振作"

        hidden = "看起来他其实在害怕被评价为'不够努力'" if "批评" in t else "无明显潜台词"
        cognitive = "过度概括" if re.search(r"(没用|废物|什么都做不好)", t) else "未见明显"

        data = {
            "emotion": emo, "valence": val, "intensity": 0.72, "confidence": 0.86,
            "risk": risk,
            "risk_signals": ["绝望/自我否定"] if risk in ("L2", "L3") else [],
            "core_need": need, "hidden": hidden, "cognitive": cognitive,
            "evidence": [t[-30:]] if t else [],
        }
        return json.dumps(data, ensure_ascii=False)

    # ---- 周报 ----
    def _weekly(self, prompt: str) -> str:
        import re as _re
        n = _re.search(r'"倾诉次数":\s*(\d+)', prompt)
        days = _re.search(r'"记录天数":\s*(\d+)', prompt)
        nv = n.group(1) if n else "几"
        dv = days.group(1) if days else "几"
        return (f"这 {dv} 天里你来了 {nv} 次，大部分话都绕不开实验和别人的评价。"
                "有两天情绪掉得比较明显，那两天的记录里都出现了‘没用’这样的词——"
                "我注意到你不是在陈述事实，更像是在给自己下结论。\n\n"
                "有个可以试试的办法：下次再冒出这个念头时，先把它写成“今天这件事我没做好”，"
                "而不是“我这个人不行”。这两句话的差别，比看上去大得多。\n\n"
                "如果掉下去的那几天越来越频繁，去心理中心聊一次会比自己扛更有用。")

    # ---- 回复 ----
    def _reply(self, prompt: str) -> str:
        if "组会" in prompt or "批评" in prompt:
            return ("组会上被批评，那种感觉不只是难过，更像是被人当众盖了个章。"
                    "你说自己'没用'的时候，我在想——是不是已经很久没人跟你说过"
                    "'你其实做得不错'了？\n\n"
                    "这周还有别的时候，也有类似的感觉吗？")
        if "睡不着" in prompt or "失眠" in prompt:
            return ("连着三天睡不好，白天还得照常去实验室坐着，这挺熬人的。"
                    "躺着的时候脑子反而最吵，那种越想睡越清醒的感觉很难受。\n\n"
                    "是躺下之后想事情停不下来，还是睡着了也容易醒？")
        if "实验" in prompt:
            return ("实验卡在那儿，最磨人的不是累，是看不到头。"
                    "每推进一步都没有反馈，久了就会开始怀疑是不是自己的问题。\n\n"
                    "现在最卡的是想不清楚方向，还是做出来和预期差太多？")
        return ("嗯，我在听。你刚才说的这些，听起来在心里压了有一阵子了。\n\n"
                "如果现在只能挑一件最想说的，你会挑哪件？")

    # ---- 输出 ----
    def _complete(self, text: str, stream: bool):
        self._json({"id": "mock", "object": "chat.completion", "model": "mock",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                 "finish_reason": "stop"}]})

    def _stream(self, text: str, stream: bool):
        if not stream:
            return self._complete(text, stream=False)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.end_headers()
        # 按字符切片，模拟真实模型的逐字吐出
        i = 0
        while i < len(text):
            step = 2 if len(text) > 80 else 1
            piece = text[i:i + step]
            i += step
            chunk = {"id": "mock", "object": "chat.completion.chunk", "model": "mock",
                     "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.012)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _json(self, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", 8900), H)
    print("Mock LLM 已启动：http://127.0.0.1:8900/v1  （仅供本地链路验证）")
    srv.serve_forever()
