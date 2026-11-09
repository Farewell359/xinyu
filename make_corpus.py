"""
心屿 XINYU —— 校园语料生成（开题阶段用的合成扩写）

对应开题汇报第 10 页"我们的对策：开源数据打底 + 线上收集的校园表达；同义改写、回译扩写，再小样本微调"。

⚠️ 说明：这里生成的是**模板合成语料**，用于把训练管线跑通、验证指标口径。
   正式阶段应替换为访谈与问卷收集的真实表达（同一 speaker 字段里标记来源，按说话人切分）。

输出：campus_corpus.jsonl  每行 {"text","label","speaker","high_stakes"}
用法：python make_corpus.py --per 120
"""
from __future__ import annotations

import argparse
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "campus_corpus.jsonl")
SEED = 42

# 每个情绪 3~4 个场景模板 × 多个表达变体，拼出校园语气
TEMPLATES: dict[str, dict[str, list]] = {
    "sadness": {
        "scene": ["实验", "论文", "成绩", "室友", "分手", "家里"],
        "subject": ["我", "我最近", "这几天我", "说真的，我"],
        "body": [
            "连着好几天提不起劲，{scene}的事一想到就闷",
            "{scene}又来一遍，我只想躺在床上什么都不做",
            "{scene}之后我一直没缓过来，晚上一个人的时候特别难受",
            "跟{scene}有关的事压得我喘不过气，觉得自己很没用",
        ],
        "tail": ["", "，也不知道跟谁说", "，说给谁听都像矫情", "，就这样耗着", "，有时候会想是不是我的问题"],
    },
    "fear": {
        "scene": ["开题", "答辩", "考试", "组会", "找工作", "保研"],
        "subject": ["我", "我最近", "这周我", "一想到"],
        "body": [
            "{scene}快到了，我越来越焦虑，晚上翻来覆去睡不着",
            "一直担心{scene}出问题，心慌得厉害",
            "{scene}的事压在心里，我不敢跟导师说，也怕自己赶不上",
            "每天醒来第一件事就是想{scene}，紧张得吃不下去",
        ],
        "tail": ["", "，头发都在掉", "，感觉自己一直在硬撑", "，越拖越不敢开始", "，谁劝都没用"],
    },
    "anger": {
        "scene": ["室友", "小组作业", "评奖", "老师", "内卷"],
        "subject": ["我真", "我真的", "说真的我", "有时候我"],
        "body": [
            "{scene}这件事让我很生气，凭什么这样",
            "碰到{scene}我就烦躁，忍不了这种不公平",
            "{scene}又来一次，我气得说不出话",
            "关于{scene}，我实在没法不介意，心里堵得慌",
        ],
        "tail": ["", "，越想越气", "，跟谁说都觉得没用", "，只能憋着", "，简直无语"],
    },
    "joy": {
        "scene": ["答辩", "实验", "面试", "比赛", "期末"],
        "subject": ["今天", "这周", "早上", "刚刚"],
        "body": [
            "{scene}终于过了，特别开心，感觉松了口气",
            "{scene}有了好结果，我居然笑出了声",
            "{scene}顺利结束，这阵子的累好像都值了",
            "{scene}被老师夸了，一整天心情都很好",
        ],
        "tail": ["", "，想请自己吃顿好的", "，走路都在飘", "，好久没这么轻松了"],
    },
    "love": {
        "scene": ["朋友", "室友", "爸妈", "喜欢的人", "同学"],
        "subject": ["我", "今天", "这几天", "说真的"],
        "body": [
            "{scene}一直陪着我，心里很暖，也有点舍不得",
            "谢谢{scene}没有嫌我烦，那种被在乎的感觉很踏实",
            "想到{scene}为我做的事，会突然有点想哭",
            "很依赖{scene}，一想到要分开就难受",
        ],
        "tail": ["", "，这种时候会觉得自己被接住了", "，想一直记着", "，原来我也能被好好对待"],
    },
    "surprise": {
        "scene": ["成绩", "导师", "朋友", "通知", "消息"],
        "subject": ["没想到", "意外的是", "我居然", "怎么会"],
        "body": [
            "{scene}的结果出乎意料，我完全没料到",
            "{scene}突然变了，我愣了好一会儿",
            "{scene}居然是这样，跟我想的完全不同",
            "{scene}的消息来得太突然，一时没反应过来",
        ],
        "tail": ["", "，现在还有点懵", "，脑子一片空白", "，不知道该高兴还是难过"],
    },
}

HIGH_STAKES_MARKERS = ["没用", "喘不过气", "撑", "翻来覆去", "心慌", "耗着", "想哭"]

# 模拟 40 位"说话人"，用于按人切分训练/测试
SPEAKERS = [f"spk-{i:02d}" for i in range(40)]


def build(per_label: int, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for label, tpl in TEMPLATES.items():
        count, seen = 0, set()
        guard = 0
        while count < per_label and guard < per_label * 40:
            guard += 1
            s = rng.choice(tpl["scene"])
            sentence = rng.choice(tpl["subject"]) + rng.choice(tpl["body"]).format(scene=s) \
                       + rng.choice(tpl["tail"])
            if sentence in seen:
                continue
            seen.add(sentence)
            rows.append({
                "text": sentence,
                "label": label,
                "speaker": rng.choice(SPEAKERS),
                "high_stakes": label in ("sadness", "fear")
                               and any(m in sentence for m in HIGH_STAKES_MARKERS),
                "source": "synthetic_template",
            })
            count += 1
    rng.shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=120, help="每个情绪类别的样本数")
    args = ap.parse_args()
    rows = build(args.per)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dist: dict[str, int] = {}
    for r in rows:
        dist[r["label"]] = dist.get(r["label"], 0) + 1
    print(f"已生成 {len(rows)} 条 → {OUT}")
    print("类别分布:", dist)
    print(f"高危样本: {sum(1 for r in rows if r['high_stakes'])} 条")
    print("⚠️ 模板合成语料仅用于跑通管线，正式阶段请替换为真实访谈语料。")


if __name__ == "__main__":
    main()
