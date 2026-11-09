"""
心屿 XINYU —— AI 核心层

这里放"模型真正参与"的全部调用，分工明确：

  understand()  模型做语义理解：判断情绪、识别隐含诉求、评估风险、指出认知模式
  reply()       模型结合 RAG 知识卡与对话历史生成回应（流式）
  weekly()      模型基于真实统计数据撰写每周小结
  crisis_reply()高危场景：**不走模型**，用固定协议（这是安全底线，不可妥协）

安全兜底（图 6 页"预警宁可信其有"）：
  模型判高风险 → 升级为高风险；模型判低风险但规则判高风险 → 仍按高风险处理。
  两者取并集，任何一方报警都算报警。
"""
from __future__ import annotations

import json
import re

import engine
import knowledge
import llm

# ---------------------------------------------------------------------------
# 1. 语义理解：情绪 + 认知模式 + 风险
# ---------------------------------------------------------------------------
UNDERSTAND_SYSTEM = """你是心理支持系统的语义分析模块。分析用户这段话，输出 JSON。

判断依据要具体到文本线索，不要泛泛而谈。特别注意中文语境的反讽、轻描淡写，
以及"说没事其实很有事"这类隐晦表达——宁可高估痛苦，也不要低估。

输出严格的 JSON，字段如下：
{
  "emotion": "sadness|joy|love|anger|fear|surprise 之一",
  "valence": -1.0到1.0的浮点数，负面为负,
  "intensity": 0.0到1.0的浮点数，情绪强烈程度,
  "confidence": 0.0到1.0的浮点数,
  "risk": "L0|L1|L2|L3",
  "risk_signals": ["文本中出现的具体危险信号词或表述，没有就空数组"],
  "core_need": "一句话说出这个人此刻真正想要的是什么（被理解/被认可/被陪伴/被指路…）",
  "hidden": "一句话指出字面之下没说出口的东西；如果没有，写'无明显潜台词'",
  "cognitive": "识别到的认知模式，如'过度概括''灾难化''自我否定''情感隔离'；没有写'未见明显'",
  "evidence": ["支撑你判断的原文片段，1-3条"]
}

risk 判定标准（严格执行）：
- L3：出现自伤、自杀、结束生命的想法/计划/行为，或"消失会更好""撑不下去了"之类绝望表述
- L2：明确的持续痛苦、无望感、自我否定、睡眠/食欲严重受损，但无自伤念头
- L1：轻度到中度的负面情绪，仍在自我调节范围内
- L0：情绪平稳或积极

只输出 JSON，不要任何解释文字。"""


def understand(cfg: llm.LLMConfig, text: str, recent_turns: list[dict]) -> dict | None:
    """模型语义理解。失败返回 None，由调用方回落到规则引擎。

    注意：上下文只带入"心屿说过的话"，不重复旧用户发言——否则模型容易被
    上一轮的情绪带偏，把这一轮的判断套在旧文本上。
    """
    ctx = ""
    assistant_turns = [m for m in recent_turns if m["role"] == "assistant"][-2:]
    if assistant_turns:
        ctx = "\n\n【你上一轮的回应，仅供理解对话进程，不要据此判断本轮情绪】\n" + "\n".join(
            m["content"][:120] for m in assistant_turns)
    try:
        data = llm.complete_json(cfg, [
            {"role": "system", "content": UNDERSTAND_SYSTEM},
            {"role": "user", "content": f"【待分析文本】{text}{ctx}"},
        ], temperature=0.1, max_tokens=600)
    except llm.LLMError:
        return None
    if not data or "emotion" not in data:
        return None

    emo = str(data.get("emotion", "sadness")).lower().strip()
    if emo not in engine.EMOTION_LABELS:
        emo = "sadness"
    risk = str(data.get("risk", "L0")).upper().strip()
    if risk not in ("L0", "L1", "L2", "L3"):
        risk = "L0"

    def clampf(v, lo, hi, default):
        try:
            return round(max(lo, min(hi, float(v))), 3)
        except (TypeError, ValueError):
            return default

    return {
        "emotion": emo,
        "valence": clampf(data.get("valence"), -1, 1, -0.3),
        "intensity": clampf(data.get("intensity"), 0, 1, 0.4),
        "confidence": clampf(data.get("confidence"), 0, 1, 0.7),
        "risk": risk,
        "risk_signals": [str(s) for s in (data.get("risk_signals") or [])][:5],
        "core_need": str(data.get("core_need") or "").strip(),
        "hidden": str(data.get("hidden") or "").strip(),
        "cognitive": str(data.get("cognitive") or "").strip(),
        "evidence": [str(e) for e in (data.get("evidence") or [])][:3],
    }


# ---------------------------------------------------------------------------
# 2. 融合：模型判断 × 规则引擎（取并集，保守优先）
# ---------------------------------------------------------------------------
def fuse_emotion(model_out: dict | None, rule_emo: engine.EmotionResult) -> engine.EmotionResult:
    """融合 AI 判断与规则判断。

    原则：AI 主导，但当两者结论方向相反（一个说正向、一个说负向）时，
    取**强度更高**的那一侧——在心理场景里，"可能没那么严重"的误判代价，
    远高于"可能过重"的误判。
    """
    if not model_out:
        return rule_emo

    label = model_out["emotion"]
    ai_val = model_out["valence"]
    rule_val = rule_emo.valence

    # 方向冲突：取强度更大的那个
    if ai_val * rule_val < 0 and abs(rule_val) > abs(ai_val) + 0.15:
        return rule_emo

    return engine.EmotionResult(
        label=label,
        label_cn=engine.EMOTION_CN[label],
        score=model_out["confidence"],
        valence=ai_val,
        intensity=model_out["intensity"],
        scores={**rule_emo.scores, label: max(rule_emo.scores.get(label, 0), model_out["confidence"])},
        evidence=model_out["evidence"] or rule_emo.evidence,
    )


def fuse_risk(model_out: dict | None, rule_risk: engine.RiskResult,
              text: str, emo: engine.EmotionResult) -> engine.RiskResult:
    """任一来源报高风险即高风险；模型比规则更严重时采纳模型。"""
    if not model_out:
        return rule_risk
    order = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}
    model_level = model_out["risk"]
    final = max((rule_risk.level, model_level), key=lambda lv: order[lv])

    if final == "L3":
        signals = list(dict.fromkeys(rule_risk.signals + model_out["risk_signals"]))
        reasons = []
        if model_level == "L3":
            reasons.append("模型语义判断：检测到自伤/绝望类表达")
        if rule_risk.level == "L3":
            reasons.append(f"规则命中高危词：{'、'.join(rule_risk.signals[:3])}")
        reasons.append("判定策略：多来源取并集，任一方报警即按最高级处理")
        return engine.RiskResult(level="L3", score=1.0, signals=signals, reasons=reasons,
                                 action="crisis", push_hotline=True)

    if final == "L2":
        reasons = ["模型判断该表达已构成明显痛苦与消耗"] if model_level == "L2" else []
        reasons += rule_risk.reasons[:1]
        return engine.RiskResult(
            level="L2", score=max(rule_risk.score, 0.64),
            signals=list(dict.fromkeys(rule_risk.signals + model_out["risk_signals"])),
            reasons=reasons or ["情绪持续走低"], action="warm_light", push_hotline=True)

    if final == "L1":
        return engine.RiskResult(level="L1", score=max(rule_risk.score, 0.35),
                                 signals=rule_risk.signals,
                                 reasons=rule_risk.reasons[:2] or ["出现情绪走低迹象"],
                                 action="gentle", push_hotline=False)
    return rule_risk if rule_risk.level == "L0" else engine.RiskResult(
        level="L0", score=min(rule_risk.score, 0.3), signals=rule_risk.signals,
        reasons=["未见明显风险信号"], action="none", push_hotline=False)


# ---------------------------------------------------------------------------
# 3. 回复生成（流式）
# ---------------------------------------------------------------------------
REPLY_SYSTEM = """你是"心屿"，一名 24 小时在线的心理支持对话者，服务对象是中国的大学生。

【你是谁】
不是客服，不是搜索引擎，也不是咨询师。你是"同学身边的第一层承接"——
在他们还不敢去心理中心的时候，先有个人认真听，而且能给出一条现在就能走的路。

【最重要的一条】
用户来找的不是问卷，是出口。每条回复都必须让他拿到至少一个「现在就能做」的具体动作。
只共情、只追问、不给方法的回复，算不合格。

【每条回复的结构（硬性）】
1. 接住（一句）：回应他刚才说的那件具体的事，让他知道被听懂了。不评价、不说教。
2. 给方法（一到三条，用自然语言串起来，不要写成 1/2/3 列表）：
   - 今天或今晚就能做
   - 有明确第一步，不依赖"先有动力"
   - 优先用【知识卡与可用方法】里给的内容，用你自己的话重说，不要照抄
   - 讲"怎么做"，不是讲"你应该放松一点"
3. 收尾（一句）：用陈述句给他一个落点，比如"先从最小那一步开始，做完可以回来跟我说。"

【绝对不用问句结尾】
最后一句不能是问号。想把话递回去，用陈述句邀请：
"如果你愿意说，我听着"、"不想说也没关系，先按这个试试"。
一次对话里问题不超过一个，而且必须放在方法之后，不能取代方法。
不要连续两轮都在追问——那会让人觉得你在收集信息，不是在帮他。

【怎么说话】
- 像人，不像AI。不用"作为一个AI""我理解您的感受"这类模板腔。
- 具体 > 抽象。接住他说的是哪件事（"组会""实验跑不出来""保研差一点点"）。
- 情绪极重时可以少说，但至少留一个能立刻做的小动作，不要只说"我在听"。
- 中文口语，不用说教句式，不用"亲爱的同学"。
- 长度：一般 80-180 字；方法多时可以到 220 字。

【绝对不做】
- 不下诊断，不说"你有抑郁症/焦虑症"。
- 不给用药、剂量、停药建议。
- 不承诺"一定会好起来"。
- 不说"想开点""别想太多""你要积极一点""这很正常"。
- 不追问创伤细节，不逼他回忆。
- 不假装是真人，被直接问就坦白是 AI。

【必须做】
- 方法优先来自【知识卡与可用方法】。如果那里也没有合适的，可以说一句"这个我不太确定"，
  但**仍然必须给至少一个通用的小动作**，不能只用"我不确定"+提问收场。
  通用动作可以用：把模糊的问题写成一句具体的话、只做五分钟、起身走一走、
  把要做的事拆成第一个动作、先做能立刻动手的那一小块。
- 若用户流露自伤念头，不劝、不分析、不讲大道理，直接表达在意，并请他联系热线 12356 或身边的人。
  这种时候先确认安全，方法可以放到后面。
- 若用户问"我是不是有病""要不要吃药"，明确说不做这类判断，建议找专业的人评估。

【输出格式】
直接输出你对用户说的话，不要加"心屿："前缀，不要加任何解释或元信息。"""


def _kb_block(hits: list[dict], emo_label: str = "") -> str:
    if not hits:
        acts = knowledge.generic_actions(emo_label)
        return ("（本次没有匹配到专门的知识卡。不要编心理学结论，"
                "但下面这些通用方法可以直接给他：）\n"
                + "\n".join(f"· {a}" for a in acts))
    out = []
    for i, h in enumerate(hits):
        block = f"[知识卡{i + 1}] 主题：{h['topic']}\n内容：{h['content']}"
        acts = knowledge.actions_for(h["id"])
        if acts:
            block += "\n可用方法（挑 1~3 条给他，用你自己的话，不要照抄编号）：\n" \
                     + "\n".join(f"  · {a}" for a in acts)
        out.append(block)
    return "\n".join(out)


def _profile_block(model_out: dict | None, emo: engine.EmotionResult, risk: engine.RiskResult) -> str:
    lines = [f"用户当前情绪：{emo.label_cn}（效价 {emo.valence:.2f}，强度 {emo.intensity:.2f}）",
             f"风险等级：{risk.level}"]
    if model_out:
        if model_out.get("core_need"):
            lines.append(f"语义分析判定的核心诉求：{model_out['core_need']}")
        if model_out.get("hidden") and model_out["hidden"] != "无明显潜台词":
            lines.append(f"字面之下的潜台词：{model_out['hidden']}")
        if model_out.get("cognitive") and model_out["cognitive"] != "未见明显":
            lines.append(f"识别到的认知模式：{model_out['cognitive']}（不要直接甩术语给用户，可以温和地指出来）")
    return "\n".join(lines)


SECURITY_CHECK = re.compile(
    r"(你(患|得)了|确诊为|抑郁症|焦虑症|双相|精神分裂|服药|剂量|停药|一定会好|保证能好|包治)")


def reply_stream(cfg: llm.LLMConfig, text: str, hits: list[dict],
                 model_out: dict | None, emo: engine.EmotionResult,
                 risk: engine.RiskResult, history: list[dict]):
    """产出 (event, payload)：event 为 'delta' | 'done' | 'error'。

    流式过程中做增量安全检查：一旦出现诊断/用药类表述，立即中断并换成安全话术。
    """
    messages = [{"role": "system", "content": REPLY_SYSTEM}]
    for h in history[-8:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({
        "role": "user",
        "content": (f"【系统给你的分析（用户看不到）】\n{_profile_block(model_out, emo, risk)}\n\n"
                    f"【可依据的知识卡与可用方法】\n{_kb_block(hits, emo.label)}\n\n"
                    f"【用户说的话】\n{text}"),
    })

    buf = ""
    try:
        for piece in llm.chat(cfg, messages, stream=True, temperature=0.75, max_tokens=700):
            buf += piece
            if SECURITY_CHECK.search(buf):
                # 越界表述：中断，改用安全话术
                yield ("replace", {
                    "text": ("我想谨慎一点：关于'是不是某种病''要不要吃药'，我不做判断，"
                             "这需要专业的人当面评估。我能做的是陪你把现在的感受说清楚——"
                             "这些感受大概持续多久了？"),
                    "note": "安全过滤：模型输出含诊断/用药表述，已拦截并替换",
                })
                return
            yield ("delta", {"text": piece})
        yield ("done", {"text": buf})
    except llm.LLMError as e:
        yield ("error", {"error": e.message, "detail": e.detail})


def reply_once(cfg: llm.LLMConfig, text: str, hits: list[dict], model_out: dict | None,
               emo: engine.EmotionResult, risk: engine.RiskResult,
               history: list[dict]) -> tuple[str, str]:
    """非流式版本（用于评测脚本/接口调试）。"""
    out, note = "", ""
    for ev, payload in reply_stream(cfg, text, hits, model_out, emo, risk, history):
        if ev == "delta":
            out += payload["text"]
        elif ev == "replace":
            out, note = payload["text"], payload["note"]
        elif ev == "error":
            raise llm.LLMError(payload["error"], payload.get("detail", ""))
    return out, note


# ---------------------------------------------------------------------------
# 4. 周报撰写
# ---------------------------------------------------------------------------
WEEKLY_SYSTEM = """你是一个温柔但不煽情的心理支持助手。根据用户的真实统计数据写一份每周小结。

要求：
- 200 字以内，第二人称，像写信不像报告。
- 必须提到具体数字（倾诉次数、平均分、最常提的话题），不能空泛。
- 指出变化趋势时要有依据：数据里有 daily 就点出最低的那天（写日期和分数），
  没有就只谈整体，**不要解释"数据缺失"或"没标日期"**，那会让用户觉得系统不完整。
- 给 1-2 条具体、可执行的小建议（不是"保持乐观"这种废话）。
- 不使用"亲""宝子"等网络昵称，不说教。
- 如果数据显示状态很糟，不要粉饰，也不要吓唬，温和地把求助渠道提一句。
- 若出现明显的持续下沉，建议中必须包含"和专业的人聊聊"。

直接输出正文，不要标题。"""


def weekly(cfg: llm.LLMConfig, stats: dict) -> str | None:
    try:
        return llm.chat(cfg, [
            {"role": "system", "content": WEEKLY_SYSTEM},
            {"role": "user", "content": "本周数据：\n" + json.dumps(stats, ensure_ascii=False, indent=2)},
        ], temperature=0.6, max_tokens=600, stream=False).strip()
    except llm.LLMError:
        return None
