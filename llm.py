"""
心屿 XINYU —— 大模型接入层（LLM Client）

支持任意 OpenAI 兼容接口（DeepSeek / 智谱 GLM / 阿里通义 / Moonshot / SiliconFlow / 本地 Ollama…）。
只用标准库 urllib 实现，支持 SSE 流式输出。

配置来源优先级：
    1. 用户在界面里保存的 config.json（data/config.json）
    2. 环境变量 XINYU_LLM_API_KEY / XINYU_LLM_BASE_URL / XINYU_LLM_MODEL

⚠️ 安全：config.json 里存的是 API Key，只落本机磁盘，不上传任何地方；
   界面上回显时永远打码。
"""
from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "data", "config.json")

# 常见服务商预设（都是 OpenAI 兼容协议）
PROVIDERS = {
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "note": "中文对话强、价格低，新用户有赠送额度",
        "apply": "https://platform.deepseek.com/api_keys",
        "candidates": ["deepseek-chat", "deepseek-reasoner"],
    },
    "zhipu": {
        "name": "智谱 GLM",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
        "note": "GLM-4-Flash 免费额度充足，课程项目首选",
        "apply": "https://open.bigmodel.cn/usercenter/apikeys",
        "candidates": ["glm-4-flash", "glm-4-flashx", "glm-4-air", "glm-4-plus"],
    },
    "dashscope": {
        "name": "阿里通义千问",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3.8-omni-flash",
        "note": "百炼平台。免费额度按模型单独开通，qwen-plus 等常报「未开通」；不确定用哪个就点「探测可用模型」",
        "apply": "https://bailian.console.aliyun.com/",
        "candidates": ["qwen3.8-omni-flash", "qwen-turbo", "qwen-plus", "qwen-max",
                       "qwen-long", "qwen-flash", "qwen2.5-7b-instruct"],
    },
    "moonshot": {
        "name": "Moonshot Kimi",
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
        "note": "长文本能力好",
        "apply": "https://platform.moonshot.cn/console/api-keys",
    },
    "siliconflow": {
        "name": "SiliconFlow 硅基流动",
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "note": "聚合多家开源模型，有免费模型",
        "apply": "https://cloud.siliconflow.cn/account/ak",
    },
    "ollama": {
        "name": "本地 Ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen2.5:7b",
        "note": "完全离线，需先 ollama pull qwen2.5:7b",
        "apply": "https://ollama.com/download",
    },
    "custom": {
        "name": "自定义",
        "base_url": "",
        "model": "",
        "note": "任意 OpenAI 兼容端点",
        "apply": "",
    },
}


@dataclass
class LLMConfig:
    provider: str = "deepseek"
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 700
    timeout: int = 45

    @property
    def ready(self) -> bool:
        if self.provider == "ollama":
            return bool(self.base_url and self.model)
        return bool(self.api_key and self.base_url and self.model)

    def masked_key(self) -> str:
        k = self.api_key or ""
        if len(k) <= 8:
            return "•" * len(k)
        return f"{k[:4]}{'•' * 10}{k[-4:]}"

    def public(self) -> dict:
        """给前端的脱敏视图，绝不回传明文 Key。"""
        return {
            "provider": self.provider,
            "provider_name": PROVIDERS.get(self.provider, {}).get("name", self.provider),
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "ready": self.ready,
            "masked_key": self.masked_key(),
            "has_key": bool(self.api_key),
        }


def _env_config() -> LLMConfig:
    provider = os.environ.get("XINYU_LLM_PROVIDER", "deepseek")
    preset = PROVIDERS.get(provider, PROVIDERS["custom"])
    return LLMConfig(
        provider=provider,
        api_key=os.environ.get("XINYU_LLM_API_KEY", ""),
        base_url=os.environ.get("XINYU_LLM_BASE_URL") or preset["base_url"],
        model=os.environ.get("XINYU_LLM_MODEL") or preset["model"],
    )


def load_config() -> LLMConfig:
    cfg = _env_config()
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                saved = json.load(f)
            for field in ("provider", "api_key", "base_url", "model",
                          "temperature", "max_tokens", "timeout"):
                if saved.get(field) not in (None, ""):
                    setattr(cfg, field, saved[field])
        except (json.JSONDecodeError, OSError):
            pass
    return cfg


def save_config(data: dict) -> LLMConfig:
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    current = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                current = json.load(f)
        except (json.JSONDecodeError, OSError):
            current = {}

    provider = data.get("provider") or current.get("provider") or "deepseek"
    preset = PROVIDERS.get(provider, PROVIDERS["custom"])
    # 换服务商时，若用户没显式给 base_url/model，就用该服务商默认值
    if data.get("provider") and data.get("provider") != current.get("provider"):
        current["base_url"] = data.get("base_url") or preset["base_url"]
        current["model"] = data.get("model") or preset["model"]

    current["provider"] = provider
    if data.get("api_key"):
        current["api_key"] = data["api_key"].strip()
    for f in ("base_url", "model"):
        if data.get(f):
            current[f] = data[f].strip()
    if data.get("temperature") is not None:
        current["temperature"] = float(data["temperature"])
    if data.get("max_tokens"):
        current["max_tokens"] = int(data["max_tokens"])

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2)
    return load_config()


def clear_key() -> None:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                cur = json.load(f)
            cur["api_key"] = ""
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cur, f, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, OSError):
            os.remove(CONFIG_PATH)


# ---------------------------------------------------------------------------
# 调用
# ---------------------------------------------------------------------------
class LLMError(Exception):
    """带可读原因的错误，直接展示给用户，而不是"出了点问题"。

    kind 用于前端区分处理方式：
        ratelimit  限流/额度 —— 等几秒可重试
        nomodel   该模型没开通/不存在 —— 必须换模型
        noauth    Key 无效 —— 必须换 Key
        balance   余额不足 —— 充值或换服务商
        server    服务商故障 —— 稍后重试
        net       网络问题
        other
    """

    def __init__(self, message: str, detail: str = "", code: int | None = None,
                 kind: str = "other"):
        super().__init__(message)
        self.message = message
        self.detail = detail
        self.code = code
        self.kind = kind

    @property
    def retryable(self) -> bool:
        return self.kind in ("ratelimit", "server", "net")

    def as_dict(self) -> dict:
        return {"error": self.message, "detail": self.detail,
                "code": self.code, "kind": self.kind,
                "retryable": self.retryable}


def _payload(cfg: LLMConfig, messages: list[dict], stream: bool,
             temperature: float | None = None, max_tokens: int | None = None,
             json_mode: bool = False) -> bytes:
    body = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature if temperature is None else temperature,
        "max_tokens": cfg.max_tokens if max_tokens is None else max_tokens,
        "stream": stream,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


def _request(cfg: LLMConfig, payload: bytes) -> urllib.request.Request:
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    return urllib.request.Request(url, data=payload, headers=headers, method="POST")


def _explain_http_error(e: urllib.error.HTTPError) -> LLMError:
    try:
        raw = e.read().decode("utf-8", "replace")
    except Exception:
        raw = ""
    detail = raw[:300]

    # 服务商通常会在 error.code / error.type 里给出真正的业务原因，
    # 光看 HTTP 状态码会把「限流」和「模型未开通」混成一句话。
    biz = ""
    try:
        obj = json.loads(raw)
        err = obj.get("error") if isinstance(obj, dict) else None
        if isinstance(err, dict):
            biz = f"{err.get('code', '')} {err.get('type', '')}".lower()
        elif isinstance(err, str):
            biz = err.lower()
    except (json.JSONDecodeError, TypeError):
        pass
    biz += " " + raw[:400].lower()

    def has(*words):
        return any(w in biz for w in words)

    # ---- 业务级分类（优先于 HTTP 状态码）----
    if has("limit_requests", "rate_limit", "rate limit", "too many requests",
           "requests rate", "throttl", "quota_exceeded", "exceeded your current request limit"):
        return LLMError(
            "触发免费额度限流（请求太频繁）",
            "这个 Key 本身是有效的，模型也有权限，只是当前 RPM/TPM 被打满了。\n"
            "· 等 10～60 秒再点「测试连接」即可；\n"
            "· 别连续狂点，会一直续上限流；\n"
            "· 若长期如此，建议换用智谱 GLM-4-Flash（免费额度宽松）。\n"
            f"原始返回：{detail}", e.code, "ratelimit")

    if has("unpurchased", "accessdenied", "not purchased", "no permission",
           "permission denied", "model.access.denied", "forbidden"):
        return LLMError(
            "该模型未开通（不是 Key 错了）",
            "这个 API Key 有效，但它名下没有开通当前模型。\n"
            "· 到百炼控制台「模型广场」确认哪些模型可调用；\n"
            "· 或在上方把模型名改成 qwen-turbo（免费额度里最常见的一个）；\n"
            "· 也可以点「自动探测可用模型」，我帮你挨个试。\n"
            f"原始返回：{detail}", e.code, "nomodel")

    if has("model not found", "model_not_found", "invalidmodel", "model not exist",
           "does not exist", "unknown model", "invalid_request_error") and e.code in (400, 404):
        return LLMError(
            "模型名不存在或接口地址不对",
            "· 检查 Base URL 是否以 /v1 结尾；\n"
            "· 检查模型名拼写（区分大小写、版本号）；\n"
            "· 或点「自动探测可用模型」。\n"
            f"原始返回：{detail}", e.code, "nomodel")

    if has("insufficient", "arrearage", "balance", "quota", "no balance"):
        return LLMError(
            "账户余额 / 免费额度已用尽",
            "到服务商控制台确认额度，或换用有免费额度的模型（如 glm-4-flash）。\n"
            f"原始返回：{detail}", e.code, "balance")

    # ---- 退回 HTTP 状态码 ----
    mapping = {
        401: ("API Key 无效或未授权", "请检查 Key 是否填错、是否已过期。", "noauth"),
        402: ("账户余额不足", "请到服务商控制台充值，或换用有免费额度的模型。", "balance"),
        403: ("没有权限访问该模型", "该模型可能未开通，请点「自动探测可用模型」。", "nomodel"),
        404: ("接口地址或模型名不存在", "请检查 Base URL 是否以 /v1 结尾、模型名是否拼写正确。", "nomodel"),
        429: ("请求太频繁或额度用尽", "稍等 10～60 秒再试，别连续点击。", "ratelimit"),
        500: ("服务商内部错误", "稍后重试即可。", "server"),
        502: ("服务商网关错误", "稍后重试即可。", "server"),
        503: ("服务商繁忙", "稍后重试即可。", "server"),
    }
    msg, hint, kind = mapping.get(
        e.code, (f"请求失败（HTTP {e.code}）", "请检查配置。", "other"))
    return LLMError(msg, f"{hint}\n原始返回：{detail}", e.code, kind)


def _explain_net_error(e: Exception) -> LLMError:
    if isinstance(e, urllib.error.URLError):
        reason = getattr(e, "reason", None)
        if isinstance(reason, socket.timeout):
            return LLMError("连接超时", "网络较慢或服务商无响应，可稍后重试或换一家服务商。",
                            None, "net")
        if isinstance(reason, socket.gaierror):
            return LLMError("无法解析服务商域名", "请检查网络连接，或确认 Base URL 拼写正确。",
                            None, "net")
        return LLMError("网络连接失败", f"请检查网络或代理设置。原始错误：{reason}", None, "net")
    if isinstance(e, socket.timeout):
        return LLMError("连接超时", "网络较慢或服务商无响应。", None, "net")
    return LLMError("调用失败", str(e), None, "other")


def chat(cfg: LLMConfig, messages: list[dict], *, stream: bool = False,
         temperature: float | None = None, max_tokens: int | None = None,
         json_mode: bool = False, retries: int = 1):
    """非流式返回完整文本；流式返回生成器（逐 delta 文本）。"""
    if not cfg.ready:
        raise LLMError("尚未配置大模型", "请在「我的 → AI 模型」里填入 API Key。")

    payload = _payload(cfg, messages, stream, temperature, max_tokens, json_mode)
    last_err: LLMError | None = None

    for attempt in range(retries + 1):
        try:
            req = _request(cfg, payload)
            resp = urllib.request.urlopen(req, timeout=cfg.timeout)
            if not stream:
                data = json.loads(resp.read())
                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    raise LLMError("返回格式异常", json.dumps(data, ensure_ascii=False)[:300])
            return _iter_stream(resp)
        except urllib.error.HTTPError as e:
            last_err = _explain_http_error(e)
            if last_err.retryable and attempt < retries:
                # 限流要比普通故障等更久，否则重试只会继续撞墙
                wait = 4.0 * (attempt + 1) if last_err.kind == "ratelimit" \
                    else 1.2 * (attempt + 1)
                time.sleep(wait)
                continue
            raise last_err
        except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
            last_err = _explain_net_error(e)
            if attempt < retries:
                time.sleep(1.0)
                continue
            raise last_err
    raise last_err or LLMError("调用失败")


def _iter_stream(resp):
    """解析 SSE 流，逐块产出增量文本。"""
    try:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
                delta = obj["choices"][0].get("delta", {})
                piece = delta.get("content") or ""
                if piece:
                    yield piece
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    finally:
        resp.close()


def complete_json(cfg: LLMConfig, messages: list[dict], *, temperature: float = 0.2,
                  max_tokens: int = 500) -> dict:
    """要求模型输出 JSON，并做健壮解析（有些模型会包 ```json 围栏）。"""
    text = chat(cfg, messages, stream=False, temperature=temperature,
                max_tokens=max_tokens, json_mode=True)
    return parse_json_loose(text)


def parse_json_loose(text: str) -> dict:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("```")[1] if "```" in t[3:] else t.strip("`")
        t = t.lstrip("json").strip() if t.lower().startswith("json") else t
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        t = t[start:end + 1]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return {}


def health_check(cfg: LLMConfig, retries: int = 0) -> dict:
    """测试连通性：发一条极短请求，返回真实结果。"""
    t0 = time.perf_counter()
    try:
        out = chat(cfg, [
            {"role": "system", "content": "你是连通性测试助手。只回复两个字：可用"},
            {"role": "user", "content": "ping"},
        ], stream=False, temperature=0, max_tokens=16, retries=retries)
        return {"ok": True, "latency_ms": round((time.perf_counter() - t0) * 1000),
                "reply": (out or "").strip()[:40], "model": cfg.model, "kind": "ok"}
    except LLMError as e:
        return {"ok": False, "latency_ms": round((time.perf_counter() - t0) * 1000),
                **e.as_dict()}


def probe_models(cfg: LLMConfig, models: list[str] | None = None) -> dict:
    """挨个试候选模型，告诉你这个 Key 到底能用哪个。

    针对的场景：Key 明明有效，但 qwen-plus / glm-4 报「未开通」，
    用户完全不知道该填什么模型名。这里逐个真实调用，返回可用清单。
    """
    if models is None:
        models = PROVIDERS.get(cfg.provider, {}).get("candidates") or \
                 [cfg.model] if cfg.model else []
    if not models:
        return {"ok": False, "error": "没有可探测的候选模型",
                "detail": "请在「模型名」里手动填写后重试。", "tried": []}

    tried = []
    winner = None
    for name in models:
        probe = LLMConfig(**{**cfg.__dict__, "model": name})
        r = health_check(probe)
        tried.append({"model": name, "ok": r.get("ok"), "kind": r.get("kind"),
                      "error": None if r.get("ok") else r.get("error"),
                      "latency_ms": r.get("latency_ms")})
        if r.get("ok"):
            winner = name
            break          # 找到能用的就停，别继续烧额度
        # 限流不代表没权限，记下来但继续试下一个
        time.sleep(0.6)

    if winner:
        return {"ok": True, "model": winner, "tried": tried,
                "message": f"这个 Key 可以用「{winner}」，已帮你选上。"}
    return {"ok": False, "tried": tried,
            "error": "候选模型都不通",
            "detail": "可能 Key 未开通任何模型，或账户欠费。建议换个服务商"
                      "（智谱 GLM-4-Flash 免费额度最宽松）。"}
