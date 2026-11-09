"""
心屿 XINYU —— 情绪识别与风险预警引擎

对应开题汇报第 6 页的三条流水线原则：
  1. 先脱敏，再进模型      -> sanitize()
  2. 回复不许自由发挥      -> knowledge.RAG 检索 + 话术模板
  3. 预警宁可信其有        -> risk.assess() 保守策略，宁可多报不可漏报

本模块不依赖任何第三方库，纯规则 + 词典实现，可直接在普通笔记本上运行。
模型层（RoBERTa 微调）见 train.py，二者接口一致，可无缝替换。
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Iterable

# ---------------------------------------------------------------------------
# 一、脱敏：匿名不是口号，是流水线的第一道工序
# ---------------------------------------------------------------------------

# 学号 / 手机号 / 身份证 / 邮箱 / 宿舍号 / 常见称谓
_SANITIZE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[邮箱]"),
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[手机号]"),
    (re.compile(r"(?<!\d)[1-9]\d{5}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"), "[身份证]"),
    (re.compile(r"(?<!\d)(20|19)\d{2}(0[1-9]|1[0-2])\d{2,6}(?!\d)"), "[学号]"),
    (re.compile(r"\d{1,2}\s*号?楼\s*\d{2,4}"), "[宿舍]"),
    (re.compile(r"[A-Za-z]:\\[^\s，。,]*"), "[路径]"),
    (re.compile(r"(我是|我叫|本人)([^\s，。,、]{1,4})(?=[，。,\s]|$)"), r"\1[姓名]"),
]

# 院系 / 专业 / 导师等可定位到小范围的身份信息
_SENSITIVE_TERMS = [
    "计算机学院", "心理学院", "文学院", "法学院", "医学院", "经管学院", "化学学院",
    "辅导员", "班主任", "导师", "导师组", "王老师", "李老师", "张老师",
]


@dataclass
class SanitizeResult:
    text: str
    hit: list[str] = field(default_factory=list)

    @property
    def has_identity(self) -> bool:
        return bool(self.hit)


def sanitize(raw: str) -> SanitizeResult:
    """剔除身份信息，返回脱敏后的文本与命中项（仅用于提示用户，不落库）。"""
    text = raw
    hit: list[str] = []
    for pattern, repl in _SANITIZE_RULES:
        if pattern.search(text):
            hit.append(pattern.pattern[:14])
            text = pattern.sub(repl, text)
    for term in _SENSITIVE_TERMS:
        if term in text:
            hit.append(term)
            text = text.replace(term, "[机构/人物]")
    return SanitizeResult(text=text, hit=sorted(set(hit)))


# ---------------------------------------------------------------------------
# 二、情绪识别：词典 + 句式特征，输出 6 类情绪与 [-1, 1] 的效价
# ---------------------------------------------------------------------------

EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

EMOTION_CN = {
    "sadness": "低落",
    "joy": "愉悦",
    "love": "温暖",
    "anger": "烦躁",
    "fear": "焦虑",
    "surprise": "惊讶",
}

# 每条词表：词 -> 权重。校园语境专属表达单独成组（对标"校园语料本地化"）
_LEXICON: dict[str, dict[str, float]] = {
    "sadness": {
        "难过": 1.0, "难受": 0.9, "压抑": 1.0, "崩溃": 1.2, "绝望": 1.6, "没意思": 0.8,
        "空虚": 0.9, "孤独": 1.0, "一个人的": 0.6, "想哭": 1.1, "哭了": 1.0, "眼泪": 0.9,
        "提不起": 0.9, "没劲": 0.7, "丧": 0.8, "低谷": 0.9, "麻木": 1.0, "没意义": 1.3,
        "厌倦": 0.8, "熬": 0.6, "撑不住": 1.3, "不想说话": 1.0, "失去": 0.9, "分手": 1.0,
        # 校园语料
        "挂科": 1.1, "延毕": 1.4, "被退学": 1.5, "保研悬": 1.3, "没进面": 1.0, "落选": 0.9,
        "组会又被批评": 1.3, "被批评": 0.9, "比赛输了": 0.9, "室友矛盾": 1.0, "被孤立": 1.2,
        "没人要我": 1.3, "投了几十份": 0.9, "简历石沉大海": 1.1,
    },
    "joy": {
        "开心": 1.0, "高兴": 1.0, "快乐": 1.0, "轻松": 0.8, "舒服": 0.7, "笑": 0.6,
        "顺利": 0.8, "通过了": 1.1, "上岸": 1.3, "拿到offer": 1.4, "拿到 offer": 1.4,
        "进步": 0.9, "满足": 0.9, "治愈": 0.9, "感谢": 0.7, "值得": 0.8, "好多了": 1.0,
        "想开了": 1.0, "有希望": 0.9, "实习过了": 1.2, "答辩过了": 1.2, "拿奖": 1.1,
        "松了一点": 1.0, "松了口气": 1.1, "轻松一点": 0.9, "好一点": 0.7, "缓和": 0.7,
        "舒服了点": 0.8, "被接住": 0.9,
    },
    "love": {
        "喜欢": 0.9, "想念": 1.0, "心疼": 0.9, "陪伴": 0.8, "谢谢": 0.6, "信任": 0.7,
        "温暖": 0.9, "依赖": 0.7, "舍不得": 0.8, "在一起": 0.7, "朋友陪": 0.9,
    },
    "anger": {
        "生气": 1.0, "烦": 0.8, "烦躁": 1.0, "讨厌": 0.9, "气死": 1.2, "不公平": 1.0,
        "凭什么": 1.0, "恶心": 1.0, "无语": 0.7, "烦死": 1.1, "忍不了": 1.1,
        "内卷": 0.8, "被针对": 1.0, "压榨": 1.0, "占座": 0.6, "吵": 0.7,
    },
    "fear": {
        "焦虑": 1.1, "害怕": 1.1, "紧张": 0.9, "担心": 0.8, "睡不着": 1.1, "失眠": 1.1,
        "心慌": 1.1, "慌": 0.8, "压力": 0.8, "怕": 0.9, "恐惧": 1.2, "不安": 0.9,
        "来不及": 1.0, "赶不上": 1.0, "卷不过": 1.0, "担心延期": 1.1, "怕延期": 1.2,
        "怕考不好": 1.1, "不敢说": 0.8, "不敢跟导师说": 1.2, "催": 0.6,
        "没睡好": 1.0, "睡不好": 1.0, "熬夜": 0.7, "紧张兮兮": 1.0,
    },
    "surprise": {
        "没想到": 1.0, "意外": 1.0, "竟然": 0.9, "居然": 0.9, "突然": 0.6, "怎么会": 0.8,
    },
}

# 否定词：出现在情绪词前 3 字内则极性翻转、强度衰减
_NEGATIONS = ["不", "没", "别", "无", "非", "未", "不是", "并不", "没那么", "一点都不"]
# 强化词
_INTENSIFIERS = {"很": 0.25, "特别": 0.3, "非常": 0.3, "太": 0.25, "超级": 0.35,
                 "巨": 0.3, "好": 0.15, "真的": 0.2, "彻底": 0.35, "一直": 0.2, "连着": 0.2}

# 高危信号：分三级，宁可信其有
_HIGH_RISK = [
    "自杀", "轻生", "不想活", "活不下去", "死了算了", "想去死", "结束自己",
    "跳楼", "割腕", "自残", "伤害自己", "吃了很多药", "吞药", "遗书", "告别",
    "撑不下去了", "没有活下去的理由", "消失了会更好", "不如死了",
]
_MID_RISK = [
    "绝望", "看不到希望", "没意义", "毫无意义", "熬不下去", "扛不住", "崩溃了",
    "活着好累", "好累", "累到", "没人需要我", "我是负担", "拖累", "麻木",
    "一直失眠", "整夜睡不着", "吃不下", "不想动", "什么都不想做", "躲起来",
]
_LOW_RISK = [
    "压力大", "焦虑", "难受", "难过", "想哭", "孤独", "委屈", "低落", "emo",
]

# 求助渠道
HOTLINES = [
    {"name": "全国 24 小时心理援助热线", "number": "12356", "note": "国家卫健委统一心理援助热线"},
    {"name": "北京心理危机研究与干预中心", "number": "010-82951332", "note": "24 小时"},
    {"name": "希望 24 热线", "number": "400-161-9995", "note": "24 小时危机干预"},
    {"name": "校园心理中心", "number": "校内预约", "note": "工作日 8:30–17:30，可线上预约"},
]


@dataclass
class EmotionResult:
    label: str
    label_cn: str
    score: float          # 该情绪置信度 0-1
    valence: float        # 效价 -1(极负面) ~ 1(极正面)
    intensity: float      # 强度 0-1
    scores: dict[str, float]
    evidence: list[str]   # 命中的证据词，用于人工复核


def analyze_emotion(text: str) -> EmotionResult:
    t = text.lower()
    scores = {k: 0.0 for k in EMOTION_LABELS}
    evidence: dict[str, list[str]] = {k: [] for k in EMOTION_LABELS}

    for label, words in _LEXICON.items():
        for word, weight in words.items():
            start = 0
            while True:
                idx = t.find(word, start)
                if idx < 0:
                    break
                w = weight
                # 前文 4 字内出现强化词
                window = t[max(0, idx - 4):idx]
                for booster, bonus in _INTENSIFIERS.items():
                    if booster in window:
                        w += bonus
                        break
                # 否定翻转（"没那么难过"）
                neg = any(n in window for n in _NEGATIONS)
                if neg:
                    w *= -0.6
                scores[label] += w
                if w > 0:
                    evidence[label].append(word)
                start = idx + len(word)

    # 标点与重复字符强化（"烦死了！！"）
    exclam = min(text.count("！") + text.count("!"), 3)
    if exclam:
        for k in ("anger", "fear", "sadness", "joy"):
            if scores[k] > 0:
                scores[k] *= 1 + 0.08 * exclam

    # 负向情绪为负值，正向为正值 → 统一 softmax 前先取正向
    pos = {k: max(0.0, v) for k, v in scores.items()}

    # 强度由原始信号量计算（在归一化之前），避免"占比 1.0 但只说了一个词"被判成极负面
    raw_total = sum(pos.values())

    # 情绪缺失兜底：无任何信号时判为轻度低落偏中性（保守，不轻易判"愉悦"）
    if raw_total < 0.15:
        return EmotionResult(
            label="sadness", label_cn="平静/未识别", score=0.35, valence=-0.05,
            intensity=0.12, scores={k: round(1 / 6, 3) for k in EMOTION_LABELS},
            evidence=["未捕获明确情绪词"],
        )

    probs = {k: v / raw_total for k, v in pos.items()}

    # 顶部情绪 + 置信度（占比与信号量的饱和函数混合，避免单词长句虚高）
    label = max(probs, key=probs.get)
    confidence = 0.45 + 0.5 * probs[label] * (1 - math.exp(-raw_total / 1.5))

    pos_sum = pos["joy"] + pos["love"] + 0.3 * pos["surprise"]
    neg_sum = pos["sadness"] + pos["fear"] + pos["anger"]
    intensity = 1 - math.exp(-(raw_total / 2.6))
    # 效价按强度收缩：低强度时不宜判到 ±1 的极端（避免"只说了句烦"被判成极度负面）
    valence = (pos_sum - neg_sum) / (pos_sum + neg_sum + 1e-6) * (0.3 + 0.7 * intensity)

    return EmotionResult(
        label=label,
        label_cn=EMOTION_CN[label],
        score=round(confidence, 3),
        valence=round(valence, 3),
        intensity=round(intensity, 3),
        scores={k: round(v, 3) for k, v in probs.items()},
        evidence=evidence[label][:5] or ["句式整体倾向"],
    )


# ---------------------------------------------------------------------------
# 三、风险分级：三级 + 保守判定
# ---------------------------------------------------------------------------

RISK_META = {
    "L0": {"level": 0, "name": "平稳", "color": "#7FA88C", "desc": "情绪状态在可自我调节范围"},
    "L1": {"level": 1, "name": "关注", "color": "#D8B36A", "desc": "出现持续低落或压力信号"},
    "L2": {"level": 2, "name": "warm", "color": "#D98F5A", "desc": "存在明显的痛苦与消耗"},
    "L3": {"level": 3, "name": "高风险", "color": "#C25B4E", "desc": "检测到自伤/绝望等高危表达"},
}


@dataclass
class RiskResult:
    level: str
    score: float
    signals: list[str]
    reasons: list[str]
    action: str        # none / gentle / warm_light / crisis
    push_hotline: bool


def assess_risk(text: str, emotion: EmotionResult, recent: Iterable[dict] | None = None) -> RiskResult:
    """保守策略：宁可多提醒一次，也绝不漏掉一个人。

    recent: 该用户最近若干条记录（含 valence / risk_level），用于识别"连续走低"。
    """
    signals: list[str] = []
    reasons: list[str] = []

    for w in _HIGH_RISK:
        if w in text:
            signals.append(w)
    if signals:
        # 高危信号直接 L3，无论情绪分值
        return RiskResult(
            level="L3", score=1.0, signals=signals,
            reasons=[f"直接命中高危表达：{'、'.join(signals[:3])}",
                     "判定策略：只要出现即按最高级处理，并立即给出求助渠道"],
            action="crisis", push_hotline=True,
        )

    mid = [w for w in _MID_RISK if w in text]
    low = [w for w in _LOW_RISK if w in text]
    signals.extend(mid + low)

    score = emotion.intensity * 0.6 + max(0.0, -emotion.valence) * 0.4
    if mid:
        score += 0.18 * min(len(mid), 3)

    # 连续走低：近 5 条均值持续为负 → 升级
    trend_note = ""
    if recent:
        vals = [r.get("valence", 0) for r in list(recent)[-5:]]
        if len(vals) >= 3 and sum(vals) / len(vals) < -0.35:
            score += 0.12
            trend_note = f"近 {len(vals)} 次倾诉平均效价 {sum(vals) / len(vals):.2f}，呈持续走低"
            reasons.append(trend_note)

    # 长时间高频：近 24 小时超过 6 条且多为负面
    if recent:
        rs = list(recent)
        neg_recent = [r for r in rs if r.get("valence", 0) < -0.2]
        if len(neg_recent) >= 5:
            score += 0.08
            reasons.append(f"近期负面倾诉累计 {len(neg_recent)} 次，频率偏高")

    score = min(score, 0.99)

    if score >= 0.62 or len(mid) >= 3:
        level, action, push = "L2", "warm_light", True
        reasons.insert(0, "多个痛苦信号叠加，建议主动把关心递过去")
    elif score >= 0.34 or mid:
        level, action, push = "L1", "gentle", False
        reasons.insert(0, "出现情绪走低迹象，保持轻触式关注")
    else:
        level, action, push = "L0", "none", False
        reasons.insert(0, "未见明显风险信号")

    if signals:
        reasons.append("命中信号：" + "、".join(dict.fromkeys(signals))[:60])

    return RiskResult(level=level, score=round(score, 3), signals=list(dict.fromkeys(signals)),
                      reasons=reasons, action=action, push_hotline=push)


def score_to_emotion_score(valence: float, intensity: float) -> int:
    """换算为用户可见的 0-100 情绪分值（越高越好）。"""
    return int(round(max(2, min(98, 50 + valence * 45 * (0.5 + 0.5 * intensity)))))


def now_ts() -> int:
    return int(time.time())


def to_dict(obj) -> dict:
    return asdict(obj)
