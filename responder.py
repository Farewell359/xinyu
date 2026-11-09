"""
心屿 XINYU —— 对话生成层

两种模式，接口一致：
  · local  ：知识库 RAG + 话术骨架（默认，断网可用，零成本）
  · llm    ：调用大模型 API（可选），但仍强制走 RAG 引用与安全过滤

安全约束（硬编码，不依赖模型自觉）：
  1. 输出先过安全检查（诊断词、医疗建议词、绝对化承诺）
  2. 高风险时无论模型说什么，都替换为危机响应模板并挂上求助渠道
  3. 每个回复都带 sources，界面上可点开看"依据哪一条"
"""
from __future__ import annotations

import json
import os
import random
import urllib.error
import urllib.request

import knowledge
from engine import EmotionResult, RiskResult, sanitize

# 不该出现在回复里的表述（心理场景里"一本正经说错话"是致命的）
_FORBIDDEN = [
    "你患有", "你得了", "确诊", "抑郁症", "焦虑症", "双相", "精神病",
    "吃药", "服药", "剂量", "建议停药", "一定会好", "保证", "包治",
    "你就是想太多", "别想那么多", "想开点就好了",
]


def _safe(text: str) -> tuple[str, list[str]]:
    hits = [w for w in _FORBIDDEN if w in text]
    if not hits:
        return text, []
    # 命中即降级为安全话术，记录命中词用于人工复核
    return ("我想谨慎一点回答你：关于'是不是某种病症''该怎么用药'这类问题，我不做判断，"
            "这需要专业的人当面评估。我能做的是陪你把现在的感受说清楚。", hits)


def _reflect(emotion: EmotionResult) -> str:
    """反映式回应：先让对方知道情绪被准确识别。"""
    return {
        "sadness": "听起来这段时间你心里压着不少东西。",
        "fear": "能感觉到你心里一直有根弦绷着。",
        "anger": "这股气堵在心里，确实很难受。",
        "joy": "这份轻松挺难得的。",
        "love": "你很在意这件事，也很在意那个人。",
        "surprise": "这件事好像有点超出你的预期。",
    }.get(emotion.label, "我听到了。")


def compose_local(text: str, emotion: EmotionResult, risk: RiskResult,
                  turn: int = 0) -> dict:
    """本地话术引擎：RAG 命中就引用，没命中就老实说'我不确定'。"""
    if risk.level == "L3":
        return {
            "reply": knowledge.CRISIS_RESPONSE,
            "sources": [{"id": "safety_crisis", "topic": "危机响应协议",
                         "content": "高危信号触发后：不解释、不劝说、不评判，直接给出可立即执行的求助动作。"}],
            "mode": "safety",
            "crisis": True,
        }

    # 越界提问（诊断 / 用药）：不查知识库，先明确边界
    if knowledge.is_out_of_scope(text):
        return {
            "reply": knowledge.OUT_OF_SCOPE["response"],
            "sources": [{"id": "scope_guard", "topic": t, "content": knowledge.OUT_OF_SCOPE["content"]}
                        for t in knowledge.OUT_OF_SCOPE["topics"]],
            "mode": "safety",
            "crisis": False,
            "note": "命中越界问题（诊断/用药）→ 固定边界话术",
        }

    hits = knowledge.retrieve(text, emotion.label)
    rng = random.Random(hash(text) & 0xFFFF)

    if not hits:
        body = knowledge.FALLBACK["response"]
        follow = knowledge.FALLBACK["follow"][0]
        sources = [{"id": "fallback", "topic": "未检索到匹配知识",
                    "content": knowledge.FALLBACK["content"]}]
        confidence_note = "这条回复没有知识库依据，仅在追问，不给建议。"
    else:
        top = hits[0]
        lead = _reflect(emotion) if turn == 0 else ""
        body = f"{lead}{top['response']}" if lead else top["response"]
        follow = rng.choice(top["follow"])
        sources = [{"id": h["id"], "topic": h["topic"], "content": h["content"],
                    "score": h["score"]} for h in hits]
        confidence_note = f"依据知识卡《{top['topic']}》（相似度 {top['score']:.2f}）"

    # 情绪走低但未到危机：轻轻把求助渠道递过去（不追问、不施压）
    if risk.level == "L2" and turn <= 1:
        body += "\n\n还有一件事我想说：你现在承受的已经不轻了，如果愿意，学校心理中心可以预约一次线下聊聊，我把入口放在下面。"
    elif turn >= 3:
        body += "\n\n（如果你愿意，我们也可以换个方向：说说最近有没有哪一刻，你觉得稍微轻一点？）"

    reply, forbidden = _safe(body)
    if forbidden:
        confidence_note += f"｜安全过滤命中：{'、'.join(forbidden)}"
    return {"reply": reply, "sources": sources, "mode": "local",
            "crisis": False, "note": confidence_note}


# ---------------------------------------------------------------------------
# 可选：大模型模式。填好环境变量 XINYU_LLM_API_KEY 后自动启用。
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是"心屿"，一名面向大学生的心理支持对话助手。严格遵循：
1. 只做支持性倾听与反映，不做任何心理疾病诊断，不给用药建议。
2. 你的回答必须基于【知识卡】里的内容，不得自行发挥医学或心理学结论。
3. 知识卡无法回答时，直接说"我不确定"，并提出一个开放式问题，不要编造。
4. 语气具体、克制、不评判，不喊口号，不说"想开点""别想太多"。
5. 每次回复不超过 120 字，先回应情绪，再最多提一个开放式问题。
6. 不使用心理咨询术语自我标榜，不承诺结果。"""


def compose_llm(text: str, emotion: EmotionResult, risk: RiskResult,
                history: list[dict] | None, turn: int = 0) -> dict | None:
    if risk.level == "L3":
        return None  # 高危一律走本地硬编码协议，不交给模型
    api_key = os.environ.get("XINYU_LLM_API_KEY")
    base = os.environ.get("XINYU_LLM_BASE_URL", "https://api.deepseek.com/v1")
    model = os.environ.get("XINYU_LLM_MODEL", "deepseek-chat")
    if not api_key:
        return None

    hits = knowledge.retrieve(text, emotion.label)
    kb_block = "\n".join(f"[知识卡{i+1}] {h['topic']}：{h['content']}" for i, h in enumerate(hits)) \
        or "（本次未检索到知识卡，请坦诚说明不确定，并追问。）"

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn_item in (history or [])[-6:]:
        msgs.append({"role": turn_item["role"], "content": turn_item["content"]})
    msgs.append({
        "role": "user",
        "content": f"【用户情绪识别】{emotion.label_cn}（效价 {emotion.valence}，强度 {emotion.intensity}）\n"
                   f"【风险等级】{risk.level}\n【知识卡】\n{kb_block}\n\n【用户倾诉】{text}",
    })

    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps({"model": model, "messages": msgs, "temperature": 0.5,
                         "max_tokens": 400}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError):
        return None

    reply, forbidden = _safe(content)
    return {
        "reply": reply,
        "sources": [{"id": h["id"], "topic": h["topic"], "content": h["content"],
                     "score": h["score"]} for h in hits]
                   or [{"id": "fallback", "topic": "未检索到匹配知识", "content": knowledge.FALLBACK["content"]}],
        "mode": "llm",
        "crisis": False,
        "note": f"模型：{model}" + (f"｜安全过滤命中：{'、'.join(forbidden)}" if forbidden else ""),
    }


def generate(text: str, emotion: EmotionResult, risk: RiskResult,
             history: list[dict] | None = None, turn: int = 0) -> dict:
    out = compose_llm(text, emotion, risk, history, turn)
    if out is None:
        out = compose_local(text, emotion, risk, turn)
    # 统一兜底：高危绝不允许被任何分支绕过
    if risk.level == "L3" and not out.get("crisis"):
        out = compose_local(text, emotion, risk, turn)
    return out
