"""
心屿 XINYU —— 服务入口

仅使用 Python 标准库，零依赖：
    python app.py            # 默认 http://127.0.0.1:8848
    python app.py --port 9000

接口一览（全部 JSON）：
    POST /api/session            获取/恢复匿名身份
    POST /api/consent            知情同意
    POST /api/chat               发送倾诉 -> 脱敏/AI理解/风险/流式回复（SSE）
    POST /api/chat/sync          同上，非流式（便于脚本调试）
    GET  /api/history            对话历史
    GET  /api/profile            情绪档案（曲线 + 概览）
    GET  /api/weekly             本周小结（AI 撰写）
    GET  /api/alerts             待查看的暖光提醒
    POST /api/alerts/ack         已读
    GET  /api/hotlines           求助渠道
    POST /api/feedback           纠正误判（回归评测集）
    GET  /api/campus             校园聚合指标（无个体数据）
    GET  /api/model              模型配置（脱敏）
    POST /api/model              保存模型配置
    POST /api/model/test         测试连通性
    POST /api/reset              清空本机数据
    GET  /api/health
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import ai
import engine
import knowledge
import llm
import moodkit
import responder
import store

ROOT = os.path.dirname(os.path.abspath(__file__))

PUBLIC = os.path.join(ROOT, "public")
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# 演示数据：让"情绪档案"一打开就有曲线可看（标 demo 标记，可一键清空）
# ---------------------------------------------------------------------------
DEMO_SCRIPT = [
    (-6, "这周连着三天没睡好，一想到实验就烦", 0.0),
    (-6, "导师今天又问进度了，我说还行，其实一点没动", -0.2),
    (-5, "组会又被批评了，说我数据太浅", -0.3),
    (-4, "今天居然跑通了一版，虽然结果不好看", 0.2),
    (-3, "室友都回家了，宿舍就我一个，特别安静，安静得有点难受", -0.35),
    (-2, "保研悬了，排名差一点点，感觉自己很没用", -0.4),
    (-1, "下午朋友叫我出去吃了碗面，走回来的时候好像松了一点", 0.15),
]


def seed_demo(conn, uid: str) -> None:
    rows = conn.execute("SELECT COUNT(*) c FROM messages WHERE user_id=?", (uid,)).fetchone()
    if rows["c"]:
        return
    import datetime as dt
    for day_offset, text, _bias in DEMO_SCRIPT:
        ts = int((dt.datetime.now() + dt.timedelta(days=day_offset)).timestamp())
        emo = engine.analyze_emotion(text)
        risk = engine.assess_risk(text, emo)
        conn.execute(
            """INSERT INTO messages(user_id, role, content, created_at, emotion, emotion_score,
                                    valence, intensity, risk_level, risk_score, signals, sources)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (uid, "user", text, ts, emo.label, emo.score, emo.valence, emo.intensity,
             risk.level, risk.score, json.dumps(risk.signals, ensure_ascii=False), "[]"))
        rep = responder.compose_local(text, emo, risk)
        conn.execute(
            """INSERT INTO messages(user_id, role, content, created_at, emotion, emotion_score,
                                    valence, intensity, risk_level, risk_score, signals, sources)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (uid, "assistant", rep["reply"], ts + 40, None, None, None, None,
             risk.level, None, "[]", json.dumps(rep["sources"], ensure_ascii=False)))
    conn.commit()


# ---------------------------------------------------------------------------
class XinyuServer(ThreadingHTTPServer):
    """客户端随手关页面/关标签页是常态，Windows 下会抛 ConnectionAbortedError。
    这类断连属于正常现象，不该在终端刷一屏 traceback 吓到演示的自己。"""

    daemon_threads = True

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionAbortedError, ConnectionResetError,
                            BrokenPipeError, TimeoutError)):
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    server_version = "Xinyu/0.1"
    _sse_open = False

    # ---- 工具 ----
    def _json(self, obj, code: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8")) or {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def _uid(self, qs: dict, body: dict | None = None) -> str:
        return (body or {}).get("uid") or (qs.get("uid", [None])[0]) or ""

    def log_message(self, fmt, *args):  # 安静一点
        if os.environ.get("XINYU_VERBOSE"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # ---- 路由 ----
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path.startswith("/api/"):
            return self._api_get(path, qs)

        # 静态文件
        if path == "/":
            path = "/index.html"
        file_path = os.path.normpath(os.path.join(PUBLIC, path.lstrip("/")))
        if not file_path.startswith(PUBLIC) or not os.path.isfile(file_path):
            self.send_error(404, "Not Found")
            return
        ctype = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        with open(file_path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _api_get(self, path: str, qs: dict):
        conn = store.connect()
        try:
            if path == "/api/health":
                cfg = llm.load_config()
                return self._json({"ok": True, "ts": engine.now_ts(),
                                   "llm": cfg.ready, "model": cfg.model if cfg.ready else None,
                                   "provider": cfg.provider if cfg.ready else None})
            if path == "/api/model":
                cfg = llm.load_config()
                return self._json({
                    "config": cfg.public(),
                    "providers": {k: {kk: vv for kk, vv in v.items()} for k, v in llm.PROVIDERS.items()},
                })
            if path == "/api/hotlines":
                return self._json({"hotlines": engine.HOTLINES})
            if path == "/api/bootstrap":
                uid = self._uid(qs)
                user = store.get_user(conn, uid) if uid else None
                return self._json({"exists": bool(user), "consent": bool(user and user["consent"])})
            if path == "/api/campus":
                return self._json(store.campus_index(conn, int(qs.get("days", ["7"])[0])))
            if path == "/api/kb":
                return self._json({"count": len(knowledge.KNOWLEDGE_BASE),
                                   "topics": [k["topic"] for k in knowledge.KNOWLEDGE_BASE]})

            uid = self._uid(qs)
            if not uid or not store.get_user(conn, uid):
                if os.environ.get("XINYU_VERBOSE"):
                    sys.stderr.write(f"[401] {path} uid={uid!r}\n")
                return self._json({"error": "未建立匿名会话"}, 401)
            if path == "/api/history":
                return self._json({"messages": store.history(conn, uid)})
            if path == "/api/profile":
                return self._json(store.profile(conn, uid, int(qs.get("days", ["30"])[0])))
            if path == "/api/weekly":
                return self._json(self._weekly(conn, uid))
            if path == "/api/alerts":
                return self._json({"alerts": store.pending_alerts(conn, uid)})
            if path == "/api/moodkit":
                return self._json(self._moodkit(conn, uid, qs))
            return self._json({"error": "not found"}, 404)
        finally:
            conn.close()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        body = self._body()
        conn = store.connect()
        try:
            if path == "/api/session":
                uid = store.ensure_user(conn, body.get("uid") or None)
                user = store.get_user(conn, uid)
                seed_demo(conn, uid)
                return self._json({"uid": uid, "consent": bool(user["consent"])})

            # ---- 模型配置：无需用户身份（本机服务，本地管理）----
            if path == "/api/model":
                try:
                    cfg = llm.save_config(body)
                except Exception as e:  # noqa: BLE001
                    return self._json({"error": f"保存失败：{e}"}, 400)
                return self._json({"ok": True, "config": cfg.public()})

            if path == "/api/model/test":
                # 允许"边填边测"：用请求里带的临时 Key，不落盘
                cfg = llm.load_config()
                if body.get("api_key"):
                    cfg.api_key = body["api_key"].strip()
                if body.get("base_url"):
                    cfg.base_url = body["base_url"].strip()
                if body.get("model"):
                    cfg.model = body["model"].strip()
                if body.get("provider"):
                    cfg.provider = body["provider"]
                return self._json(llm.health_check(cfg))

            if path == "/api/model/probe":
                # 挨个试候选模型，找出这个 Key 到底能用哪个（限流/未开通分开报）
                cfg = llm.load_config()
                for f in ("api_key", "base_url", "model"):
                    if body.get(f):
                        setattr(cfg, f, body[f].strip())
                if body.get("provider"):
                    cfg.provider = body["provider"]
                models = body.get("models") or None
                return self._json(llm.probe_models(cfg, models))

            uid = self._uid({}, body)
            if not uid or not store.get_user(conn, uid):
                return self._json({"error": "未建立匿名会话"}, 401)

            if path == "/api/consent":
                store.set_consent(conn, uid, bool(body.get("consent")))
                return self._json({"ok": True})

            if path == "/api/chat":
                return self._chat(conn, uid, body, stream=True)

            if path == "/api/chat/sync":
                return self._chat(conn, uid, body, stream=False)

            if path == "/api/alerts/ack":
                store.ack_alert(conn, uid, int(body.get("id", 0)))
                return self._json({"ok": True})

            if path == "/api/feedback":
                store.add_feedback(conn, uid, body.get("message_id"), body.get("judged", ""),
                                   body.get("corrected", ""), body.get("note", ""))
                return self._json({"ok": True, "total": store.feedback_count(conn)})

            if path == "/api/reset":
                conn.execute("DELETE FROM messages WHERE user_id=?", (uid,))
                conn.execute("DELETE FROM alerts WHERE user_id=?", (uid,))
                conn.execute("DELETE FROM feedbacks WHERE user_id=?", (uid,))
                conn.commit()
                return self._json({"ok": True})

            return self._json({"error": "not found"}, 404)
        finally:
            conn.close()

    # ---- 核心：一条消息走完的流程（AI 主导，规则兜底）----
    def _chat(self, conn, uid: str, body: dict, stream: bool = True):
        raw = (body.get("text") or "").strip()
        if not raw:
            return self._json({"error": "内容为空"}, 400)
        if len(raw) > 1000:
            return self._json({"error": "一次最多 1000 字，可以分段说"}, 400)

        t0 = time.perf_counter()

        # ---------- 第一阶段：本地即时分析（不依赖网络）----------
        san = engine.sanitize(raw)
        clean = san.text
        rule_emo = engine.analyze_emotion(clean)
        recent = store.recent_user_messages(conn, uid, 8)
        rule_risk = engine.assess_risk(clean, rule_emo, recent)
        hist = [{"role": "user" if m["role"] == "user" else "assistant",
                 "content": m["content"]} for m in recent[-6:]]
        turn = sum(1 for m in recent if m["role"] == "user")

        cfg = llm.load_config()
        ai_on = cfg.ready

        # ---------- 第二阶段：AI 语义理解 ----------
        model_out = None
        ai_error = None
        if ai_on:
            try:
                model_out = ai.understand(cfg, clean, hist)
                if model_out is None:
                    ai_error = "语义分析返回格式异常，已使用本地引擎结果"
            except Exception as e:  # noqa: BLE001
                ai_error = f"语义分析调用失败：{e}"

        emo = ai.fuse_emotion(model_out, rule_emo)
        risk = ai.fuse_risk(model_out, rule_risk, clean, emo)

        # 知识卡检索：AI 模式下按模型判定的情绪重新检索，命中率更高
        hits = knowledge.retrieve(clean, emo.label)

        # ---------- 第三阶段：生成 ----------
        if risk.level == "L3":
            # 高危：绝不交给模型自由发挥，走固定协议
            rep = responder.compose_local(clean, emo, risk, turn)
            mode, note = "safety", "危机响应协议（硬编码，模型无权覆盖）"
        elif ai_on:
            mode, note = "ai", ""
        else:
            rep = responder.generate(clean, emo, risk, hist, turn)
            mode, note = "local", rep.get("note", "")

        meta = self._chat_meta(uid, clean, san, emo, risk, hits, model_out,
                               rule_emo, rule_risk, mode, note, ai_on, ai_error, turn)

        # ---------- 第四阶段：落库（先存用户消息，回复存完再返回）----------
        def persist(reply_text: str):
            with _LOCK:
                msg_id = store.add_message(
                    conn, uid, "user", clean,
                    emotion=emo.label, emotion_score=emo.score, valence=emo.valence,
                    intensity=emo.intensity, risk_level=risk.level, risk_score=risk.score,
                    signals=risk.signals, sources=hits)
                store.add_message(conn, uid, "assistant", reply_text,
                                  risk_level=risk.level, sources=hits)
                alert = None
                if risk.level in ("L2", "L3"):
                    reason = "；".join(risk.reasons[:2])
                    aid = store.add_alert(conn, uid, risk.level, reason)
                    alert = {
                        "id": aid, "level": risk.level,
                        "message": knowledge.WARM_LIGHT_MESSAGE if risk.level == "L2"
                        else {"title": "现在需要有人陪着你",
                              "body": "你刚才说的是很危险的信号，我不想把它轻轻带过。请允许我们先把求助渠道递到你手上。",
                              "cta": ["我现在就打", "再陪我说两句"]},
                        "hotlines": engine.HOTLINES,
                    }
            return msg_id, alert

        if not stream:
            if mode == "ai":
                try:
                    reply_text, note2 = ai.reply_once(cfg, clean, hits, model_out, emo, risk, hist)
                    note = note2 or note
                except llm.LLMError as e:
                    rep = responder.generate(clean, emo, risk, hist, turn)
                    reply_text = rep["reply"]
                    mode = "local"
                    ai_error = f"{e.message}｜已降级为本地知识库"
            else:
                reply_text = rep["reply"]
            msg_id, alert = persist(reply_text)
            meta.update({"message_id": msg_id, "reply": reply_text, "alert": alert,
                         "note": note, "mode": mode, "ai_error": ai_error,
                         "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})
            return self._json(meta)

        # ---------- 流式返回（SSE）----------
        if mode != "ai":
            # 本地/协议模式：整段吐出
            msg_id, alert = persist(rep["reply"])
            meta.update({"message_id": msg_id, "alert": alert, "sources": rep["sources"]})
            self._sse_start()
            self._sse_send("meta", meta)
            self._sse_send("delta", {"text": rep["reply"]})
            self._sse_send("done", {"text": rep["reply"],
                                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})
            self._sse_end()
            return

        # AI 模式：先发 meta，再逐字推送
        self._sse_start()
        meta_for_client = dict(meta)
        self._sse_send("meta", meta_for_client)

        acc, replaced = "", ""
        try:
            for ev, payload in ai.reply_stream(cfg, clean, hits, model_out, emo, risk, hist):
                if not self._sse_open:
                    # 前端已断开（关页面/切走）就别继续拉流了，省 token
                    return
                if ev == "delta":
                    acc += payload["text"]
                    self._sse_send("delta", payload)
                elif ev == "replace":
                    acc, replaced = payload["text"], payload.get("note", "")
                    self._sse_send("replace", payload)
                elif ev == "error":
                    raise llm.LLMError(payload["error"], payload.get("detail", ""))
        except llm.LLMError as e:
            # 模型失败 → 降级本地，并把真实原因告诉用户（不装作没事）
            fb = responder.generate(clean, emo, risk, hist, turn)
            acc = fb["reply"]
            kind_hint = {"ratelimit": "（额度限流，稍后再试）",
                         "nomodel": "（该模型未开通，试试「探测可用模型」）",
                         "noauth": "（Key 无效）",
                         "balance": "（额度用尽）"}.get(getattr(e, "kind", ""), "")
            ai_error = f"{e.message}{kind_hint}｜已降级为本地知识库"
            self._sse_send("fallback", {"error": e.message, "detail": e.detail,
                                        "kind": getattr(e, "kind", "other"),
                                        "retryable": getattr(e, "retryable", False),
                                        "text": acc, "mode": "local"})

        if not acc.strip():
            acc = "我在。刚才有点卡住了，能再说一句吗？"

        msg_id, alert = persist(acc)
        self._sse_send("done", {
            "text": acc, "message_id": msg_id, "alert": alert,
            "note": replaced or note, "ai_error": ai_error,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        })
        self._sse_end()

    # ---- 分析结果打包 ----
    def _chat_meta(self, uid, clean, san, emo, risk, hits, model_out,
                   rule_emo, rule_risk, mode, note, ai_on, ai_error, turn):
        return {
            "sanitized": clean,
            "sanitize_hits": san.hit,
            "emotion": {
                "label": emo.label, "label_cn": emo.label_cn, "score": emo.score,
                "valence": emo.valence, "intensity": emo.intensity,
                "scores": emo.scores, "evidence": emo.evidence,
            },
            "risk": {
                "level": risk.level, "score": risk.score, "signals": risk.signals,
                "reasons": risk.reasons, "action": risk.action,
                "meta": engine.RISK_META[risk.level],
            },
            # AI 的语义洞察（用户可见，这是"AI 真的在理解"的证据）
            "insight": {
                "core_need": (model_out or {}).get("core_need", ""),
                "hidden": (model_out or {}).get("hidden", ""),
                "cognitive": (model_out or {}).get("cognitive", ""),
                "evidence": (model_out or {}).get("evidence", []),
            },
            "sources": [{"id": h["id"], "topic": h["topic"], "content": h["content"],
                         "score": h["score"]} for h in hits],
            "mode": mode,
            "note": note,
            "ai_enabled": ai_on,
            "ai_error": ai_error,
            # 双引擎对照：让"AI 主导、规则兜底"这个设计看得见
            "engines": {
                "rule": {"emotion": rule_emo.label_cn, "risk": rule_risk.level},
                "ai": ({"emotion": engine.EMOTION_CN.get(model_out["emotion"], ""),
                        "risk": model_out["risk"]} if model_out else None),
            },
            "crisis": risk.level == "L3",
            "emotion_score": engine.score_to_emotion_score(emo.valence, emo.intensity),
        }

    # ---- SSE 工具 ----
    def _sse_start(self):
        # SSE 是一次性长响应：必须明确告知客户端「读完就断」。
        # 之前写 keep-alive 又没给 Content-Length，浏览器无法判断边界，
        # 会一直等下一个字节 → 前端 finally 不执行 → 按钮永久 disabled，
        # 表现为「刷新后只能发一条」。这里改为 close 并置位 close_connection。
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self._sse_open = True

    def _sse_send(self, event: str, data: dict):
        # 用户关页面 / 切走时连接会断，Windows 抛 ConnectionAbortedError，
        # 只捕获 BrokenPipe 不够，会刷一屏 traceback。
        if not self._sse_open:
            return
        try:
            chunk = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            self.wfile.write(chunk.encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            self._sse_open = False

    def _sse_end(self):
        if not self._sse_open:
            return
        try:
            self.wfile.write(b"event: end\ndata: {}\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        self._sse_open = False

    # ---- 心情补给（行为激活处方）----
    def _moodkit(self, conn, uid: str, qs: dict) -> dict:
        prof = store.profile(conn, uid)
        points = prof.get("points") or []
        # 最近一次的风险等级：没有记录就当作 L0
        latest_risk = points[-1]["risk_level"] if points else "L0"
        avg = (prof.get("summary") or {}).get("avg")
        # "换一批"：用时间戳做种子，避免每次刷新都变
        seed = int(qs.get("seed", ["0"])[0]) or None
        return moodkit.prescription(avg, latest_risk, seed)

    # ---- 周报（AI 撰写 + 规则统计）----
    def _weekly(self, conn, uid: str) -> dict:
        base = store.weekly_report(conn, uid)
        cfg = llm.load_config()
        if not cfg.ready or not base["total"]:
            base["narrative"] = None
            base["narrative_source"] = "rule"
            return base
        try:
            stats = {
                "统计区间": base["range"],
                "记录天数": base["days"],
                "倾诉次数": base["total"],
                "平均情绪分": base["avg"],
                "明显下沉次数": base["l2plus"],
                "高频话题": base["topics"],
                "情绪分布": base["distribution"],
            }
            text = ai.weekly(cfg, stats)
            base["narrative"] = text
            base["narrative_source"] = "ai" if text else "rule"
        except Exception:  # noqa: BLE001
            base["narrative"] = None
            base["narrative_source"] = "rule"
        return base


def main():
    ap = argparse.ArgumentParser(description="心屿 XINYU · AI 心理树洞与情绪预警")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8848)))
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    args = ap.parse_args()

    store.connect().close()
    srv = XinyuServer((args.host, args.port), Handler)
    cfg = llm.load_config()
    mode = f"AI 驱动（{llm.PROVIDERS.get(cfg.provider, {}).get('name', cfg.provider)} / {cfg.model}）" \
        if cfg.ready else "本地知识库（尚未配置模型，界面里可填 API Key）"
    print("=" * 62)
    print("  心屿 XINYU · 先预警，再陪伴")
    print(f"  地址: http://{args.host}:{args.port}")
    print(f"  对话: {mode}")
    print(f"  知识卡: {len(knowledge.KNOWLEDGE_BASE)} 条   数据: {store.DB_PATH}")
    print("=" * 56)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
        srv.server_close()


if __name__ == "__main__":
    main()
