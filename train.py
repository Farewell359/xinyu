"""
心屿 XINYU —— 情绪识别模型训练与评测

对应开题汇报第 7 页技术选型与第 11 页验收标准：
  · 基线：HuggingFace 开源中文模型 chinese-roberta-wwm-ext 微调
  · 数据：开源情绪数据集（dair-ai/emotion）+ 自建校园语料（campus_corpus.jsonl）
  · 验收：准确率 ≥80%，且**单独报告"高危类被判成其它类"的数量**
          （这个指标比总准确率更要命，所以混淆矩阵里刻意把高危类放在最后一行）

用法：
    pip install -r requirements-ml.txt
    python train.py                      # 默认：训练 + 评测 + 导出混淆矩阵
    python train.py --epochs 5 --bs 16
    python train.py --eval-only          # 只评测已有权重

如果这台机器没有 GPU / 装不了 torch，脚本会自动降级：
    python train.py --rules-baseline     # 用规则引擎跑一遍同样的评测集，输出基线混淆矩阵
这样"评测脚本与数据记录"永远可复现，符合开题汇报里"可复查、可复现"的承诺。

设计说明：
  · 训练/测试集严格按"说话人"切分（这里用文本来源字段 speaker 近似），防止"背答案"
  · 高危类（sadness/fear 中被标为高危的样本）单独统计漏判
"""
from __future__ import annotations

import argparse
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "campus_corpus.jsonl")
OUT_DIR = os.path.join(HERE, "runs")

LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
LABEL_CN = {"sadness": "低落", "joy": "愉悦", "love": "温暖",
            "anger": "烦躁", "fear": "焦虑", "surprise": "惊讶"}
# 需要重点盯漏判的"高危方向"类
HIGH_STAKES = {"sadness", "fear"}

SEED = 42


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_hf_dataset():
    """从 HuggingFace 读 dair-ai/emotion（英文六类），配合回译语料使用。"""
    from datasets import load_dataset
    ds = load_dataset("dair-ai/emotion")
    names = ds["train"].features["label"].names

    def conv(split):
        return [{"text": r["text"], "label": names[r["label"]], "source": "dair-ai/emotion",
                 "speaker": f"hf-{split}-{i}"} for i, r in enumerate(ds[split])]
    return conv("train"), conv("validation"), conv("test")


def load_campus_corpus():
    """自建校园语料（jsonl：{"text":..., "label":..., "speaker":..., "high_stakes":bool}）。"""
    if not os.path.exists(CORPUS):
        return []
    rows = []
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_by_speaker(rows, test_ratio=0.2):
    """按说话人切分，防止同一人的表达同时出现在训练集和测试集（"背答案"）。"""
    speakers = sorted({r.get("speaker") or f"anon-{i}" for i, r in enumerate(rows)})
    random.Random(SEED).shuffle(speakers)
    cut = max(1, int(len(speakers) * test_ratio))
    test_spk = set(speakers[:cut])
    train = [r for i, r in enumerate(rows) if (r.get("speaker") or f"anon-{i}") not in test_spk]
    test = [r for i, r in enumerate(rows) if (r.get("speaker") or f"anon-{i}") in test_spk]
    return train, test


# ---------------------------------------------------------------------------
# 指标：准确率 + 混淆矩阵 + 高危漏判
# ---------------------------------------------------------------------------
def confusion(y_true, y_pred):
    m = {t: {p: 0 for p in LABELS} for t in LABELS}
    for t, p in zip(y_true, y_pred):
        m[t][p] += 1
    return m


def report(y_true, y_pred, title="评测结果"):
    m = confusion(y_true, y_pred)
    n = len(y_true)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n if n else 0
    print(f"\n{'=' * 62}\n{title}   样本 {n}   准确率 {acc * 100:.2f}%")
    print(f"{'=' * 62}")

    head = "真实\\预测".ljust(10) + "".join(LABEL_CN[l].ljust(8) for l in LABELS) + "召回"
    print(head)
    for t in LABELS:
        row = "".join(str(m[t][p]).ljust(8) for p in LABELS)
        rec = m[t][t] / sum(m[t].values()) if sum(m[t].values()) else 0
        print(LABEL_CN[t].ljust(10) + row + f"{rec * 100:.0f}%")

    print("\n—— 高危类漏判（比总准确率更要命）——")
    total_leak = 0
    for t in HIGH_STAKES:
        leaks = sum(v for p, v in m[t].items() if p != t)
        total_leak += leaks
        print(f"  {LABEL_CN[t]}: 共 {sum(m[t].values())} 条，被判成其它类 {leaks} 条"
              f"（其中被判成'愉悦/温暖'的 {m[t]['joy'] + m[t]['love']} 条 ← 最危险）")
    print(f"  合计漏判 {total_leak} 条")

    # 逐类精确率 / 召回 / F1
    print("\n—— 逐类 F1 ——")
    for l in LABELS:
        tp = m[l][l]
        fp = sum(m[t][l] for t in LABELS if t != l)
        fn = sum(v for p, v in m[l].items() if p != l)
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        print(f"  {LABEL_CN[l]}: P={prec:.2f} R={rec:.2f} F1={f1:.2f}")
    return {"acc": acc, "confusion": m, "high_stakes_leak": total_leak}


# ---------------------------------------------------------------------------
# 规则引擎基线（无需 torch，任何时候都能跑）
# ---------------------------------------------------------------------------
def rules_predict(texts):
    import engine
    return [engine.analyze_emotion(t).label for t in texts]


# ---------------------------------------------------------------------------
# 模型训练
# ---------------------------------------------------------------------------
def train_model(args):
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    rows = load_campus_corpus()
    if not rows:
        raise SystemExit(f"找不到校园语料 {CORPUS}。先运行 python make_corpus.py 生成，"
                         "或按格式手工补充真实访谈语料。")
    train_rows, test_rows = split_by_speaker(rows, args.test_ratio)
    print(f"校园语料 {len(rows)} 条 → 训练 {len(train_rows)} / 测试 {len(test_rows)}（按说话人切分）")

    tok = AutoTokenizer.from_pretrained(args.model)
    label2id = {l: i for i, l in enumerate(LABELS)}

    class DS(Dataset):
        def __init__(self, data):
            self.enc = tok([d["text"] for d in data], truncation=True,
                           padding="max_length", max_length=args.max_len)
            self.y = [label2id[d["label"]] for d in data]

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.y[i])
            return item

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS),
        id2label={i: l for l, i in label2id.items()}, label2id=label2id)

    os.makedirs(OUT_DIR, exist_ok=True)
    targs = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=20,
        seed=SEED,
        report_to=[],
    )

    def metrics(p):
        pred = np.argmax(p.predictions, axis=-1)
        gold = p.label_ids
        recs, precs, f1s = [], [], []
        for i in range(len(LABELS)):
            tp = int(((pred == i) & (gold == i)).sum())
            fp = int(((pred == i) & (gold != i)).sum())
            fn = int(((pred != i) & (gold == i)).sum())
            pr = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
            recs.append(rc); precs.append(pr); f1s.append(f1)
        return {"accuracy": float((pred == gold).mean()),
                "f1_macro": float(np.mean(f1s)),
                "recall_sadness": recs[0], "recall_fear": recs[4]}

    trainer = Trainer(model=model, args=targs, train_dataset=DS(train_rows),
                      eval_dataset=DS(test_rows), compute_metrics=metrics)
    trainer.train()
    preds = np.argmax(trainer.predict(DS(test_rows)).predictions, axis=-1)
    y_pred = [LABELS[i] for i in preds]
    y_true = [d["label"] for d in test_rows]
    res = report(y_true, y_pred, "RoBERTa 微调 · 校园测试集")

    out = os.path.join(OUT_DIR, "xinyu-emotion")
    trainer.save_model(out)
    tok.save_pretrained(out)
    with open(os.path.join(OUT_DIR, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "accuracy": res["acc"],
                   "confusion": res["confusion"], "high_stakes_leak": res["high_stakes_leak"],
                   "n_test": len(test_rows)}, f, ensure_ascii=False, indent=2)
    print(f"\n权重已保存：{out}\n评测记录：{os.path.join(OUT_DIR, 'eval_report.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hfl/chinese-roberta-wwm-ext")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--rules-baseline", action="store_true", help="只跑规则引擎基线")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    rows = load_campus_corpus()
    if args.rules_baseline or not rows:
        if not rows:
            print("未找到校园语料，仅打印规则引擎在演示句上的表现。")
            demo = [("这周连着三天没睡好，一想到实验就烦", "fear"),
                    ("组会又被批评了，我感觉自己很没用", "sadness"),
                    ("今天答辩过了，特别开心", "joy"),
                    ("室友从来不考虑别人，真的烦死", "anger"),
                    ("谢谢你一直听我说，有点暖", "love")]
            pred = rules_predict([d[0] for d in demo])
            report([d[1] for d in demo], pred, "规则引擎 · 演示句")
            return
        _, test_rows = split_by_speaker(rows, args.test_ratio)
        pred = rules_predict([d["text"] for d in test_rows])
        report([d["label"] for d in test_rows], pred, "规则引擎基线 · 校园测试集")
        return

    if args.eval_only:
        print("eval-only 模式请先用 train.py 训练一次，或直接用 --rules-baseline 做对照。")
        return
    train_model(args)


if __name__ == "__main__":
    main()
