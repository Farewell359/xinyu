"""
心屿 XINYU —— 数据层（SQLite）

设计要点（对应开题汇报第 8 页"隐私默认匿名"）：
  · 不存姓名、手机号、学号；用户标识只是一个本地随机 UUID，可在设置里重置
  · 消息正文落库前已完成脱敏
  · 没有任何"上报"字段，也没有管理员查个体数据的接口
  · 聚合指标只统计整体均值与条数，且样本不足时拒绝输出
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import uuid

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "xinyu.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL,
    consent INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    emotion TEXT,
    emotion_score REAL,
    valence REAL,
    intensity REAL,
    risk_level TEXT,
    risk_score REAL,
    signals TEXT,
    sources TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id)
);
CREATE INDEX IF NOT EXISTS idx_msg_user_time ON messages(user_id, created_at);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    level TEXT NOT NULL,
    reason TEXT,
    created_at INTEGER NOT NULL,
    acknowledged INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS feedbacks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    message_id INTEGER,
    judged_emotion TEXT,
    corrected_emotion TEXT,
    note TEXT,
    created_at INTEGER NOT NULL
);
"""


def connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def ensure_user(conn: sqlite3.Connection, user_id: str | None) -> str:
    uid = user_id or uuid.uuid4().hex
    conn.execute("INSERT OR IGNORE INTO users(id, created_at, consent) VALUES(?,?,0)",
                 (uid, int(dt.datetime.now().timestamp())))
    conn.commit()
    return uid


def set_consent(conn: sqlite3.Connection, user_id: str, consent: bool) -> None:
    conn.execute("UPDATE users SET consent=? WHERE id=?", (1 if consent else 0, user_id))
    conn.commit()


def get_user(conn: sqlite3.Connection, user_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return dict(row) if row else None


def add_message(conn: sqlite3.Connection, user_id: str, role: str, content: str,
                **meta) -> int:
    cur = conn.execute(
        """INSERT INTO messages(user_id, role, content, created_at, emotion, emotion_score,
                                valence, intensity, risk_level, risk_score, signals, sources)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
        (user_id, role, content, int(dt.datetime.now().timestamp()),
         meta.get("emotion"), meta.get("emotion_score"), meta.get("valence"),
         meta.get("intensity"), meta.get("risk_level"), meta.get("risk_score"),
         json.dumps(meta.get("signals") or [], ensure_ascii=False),
         json.dumps(meta.get("sources") or [], ensure_ascii=False)),
    )
    conn.commit()
    return int(cur.lastrowid)


def recent_user_messages(conn: sqlite3.Connection, user_id: str, limit: int = 12) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM messages WHERE user_id=? AND role='user' ORDER BY id DESC LIMIT ?",
        (user_id, limit)).fetchall()
    out = []
    for r in reversed(rows):
        d = dict(r)
        d["signals"] = json.loads(d["signals"] or "[]")
        d["sources"] = json.loads(d["sources"] or "[]")
        out.append(d)
    return out


def history(conn: sqlite3.Connection, user_id: str, limit: int = 200) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM messages WHERE user_id=? ORDER BY id ASC LIMIT ?",
        (user_id, limit)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["signals"] = json.loads(d["signals"] or "[]")
        d["sources"] = json.loads(d["sources"] or "[]")
        out.append(d)
    return out


def add_alert(conn: sqlite3.Connection, user_id: str, level: str, reason: str) -> int:
    cur = conn.execute("INSERT INTO alerts(user_id, level, reason, created_at) VALUES(?,?,?,?)",
                       (user_id, level, reason, int(dt.datetime.now().timestamp())))
    conn.commit()
    return int(cur.lastrowid)


def pending_alerts(conn: sqlite3.Connection, user_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM alerts WHERE user_id=? AND acknowledged=0 ORDER BY id DESC",
        (user_id,)).fetchall()
    return [dict(r) for r in rows]


def ack_alert(conn: sqlite3.Connection, user_id: str, alert_id: int) -> None:
    conn.execute("UPDATE alerts SET acknowledged=1 WHERE id=? AND user_id=?", (alert_id, user_id))
    conn.commit()


def add_feedback(conn: sqlite3.Connection, user_id: str, message_id: int | None,
                 judged: str, corrected: str, note: str = "") -> None:
    conn.execute(
        "INSERT INTO feedbacks(user_id, message_id, judged_emotion, corrected_emotion, note, created_at)"
        " VALUES(?,?,?,?,?,?)",
        (user_id, message_id, judged, corrected, note, int(dt.datetime.now().timestamp())))
    conn.commit()


def feedback_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) c FROM feedbacks").fetchone()["c"]


# ---------------------------------------------------------------------------
# 情绪档案：曲线 / 周报
# ---------------------------------------------------------------------------

def profile(conn: sqlite3.Connection, user_id: str, days: int = 30) -> dict:
    since = int((dt.datetime.now() - dt.timedelta(days=days)).timestamp())
    rows = conn.execute(
        """SELECT created_at, valence, intensity, emotion, risk_level FROM messages
           WHERE user_id=? AND role='user' AND created_at>=? ORDER BY created_at ASC""",
        (user_id, since)).fetchall()

    points = []
    by_day: dict[str, list[dict]] = {}
    for r in rows:
        v = r["valence"] or 0
        it = r["intensity"] or 0
        score = int(round(max(2, min(98, 50 + v * 45 * (0.5 + 0.5 * it)))))
        day = dt.datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d")
        item = {"ts": r["created_at"], "score": score, "valence": v,
                "emotion": r["emotion"], "risk_level": r["risk_level"]}
        points.append(item)
        by_day.setdefault(day, []).append(item)

    daily = [{"date": d, "score": round(sum(x["score"] for x in v) / len(v), 1),
              "count": len(v),
              "emotion": max({e: sum(1 for x in v if x["emotion"] == e)
                              for e in {x["emotion"] for x in v}}.items(),
                             key=lambda kv: kv[1])[0]}
             for d, v in sorted(by_day.items())]

    # 连续使用天数
    streak, cursor = 0, dt.date.today()
    dayset = set(by_day)
    while cursor.isoformat() in dayset:
        streak += 1
        cursor -= dt.timedelta(days=1)

    scores = [p["score"] for p in points]
    distinct_topics = len({r["emotion"] for r in rows})
    return {
        "points": points,
        "daily": daily,
        "summary": {
            "total": len(points),
            "avg": round(sum(scores) / len(scores), 1) if scores else None,
            "best": max(scores) if scores else None,
            "worst": min(scores) if scores else None,
            "streak": streak,
            "emotion_kinds": distinct_topics,
            "trend": _trend(scores),
        },
    }


def _trend(scores: list[int]) -> str:
    if len(scores) < 4:
        return "insufficient"
    n = max(3, len(scores) // 2)
    front = sum(scores[:n]) / n
    back = sum(scores[-n:]) / n
    diff = back - front
    if diff > 6:
        return "up"
    if diff < -6:
        return "down"
    return "flat"


WEEKLY_ITEMS = [
    ("疲惫", ["累", "熬", "撑", "失眠", "睡不着", "没力气"]),
    ("学业", ["实验", "论文", "考试", "保研", "组会", "答辩", "绩点", "延期"]),
    ("关系", ["室友", "朋友", "分手", "喜欢", "孤独", "导师", "家人"]),
    ("自我评价", ["不行", "没用", "废物", "比不上", "讨厌自己"]),
    ("期待", ["希望", "想", "打算", "上岸", "拿到"]),
]


def weekly_report(conn: sqlite3.Connection, user_id: str) -> dict:
    now = dt.datetime.now()
    start = now - dt.timedelta(days=7)
    rows = conn.execute(
        """SELECT * FROM messages WHERE user_id=? AND role='user' AND created_at>=?
           ORDER BY created_at ASC""",
        (user_id, int(start.timestamp()))).fetchall()

    texts = [r["content"] for r in rows]
    keyword_stats = []
    for label, kws in WEEKLY_ITEMS:
        cnt = sum(1 for t in texts if any(k in t for k in kws))
        if cnt:
            keyword_stats.append({"label": label, "count": cnt})
    keyword_stats.sort(key=lambda x: -x["count"])

    scores = [int(round(max(2, min(98, 50 + (r["valence"] or 0) * 45)))) for r in rows]
    avg = round(sum(scores) / len(scores), 1) if scores else None
    l2plus = sum(1 for r in rows if (r["risk_level"] or "L0") >= "L2")

    if not rows:
        headline = "这一周还没有留下记录。想说话的时候，随时来。"
    elif avg is not None and avg >= 62:
        headline = "这一周整体是明亮的。有些时刻被你自己撑住了，也值得记下来。"
    elif avg is not None and avg >= 45:
        headline = "这一周起伏不小——有几天明显更沉。这不是你的错，情绪本来就有波谷。"
    else:
        headline = "这一周你扛得挺辛苦。记录的这些，说明你没有真的放弃自己。"

    suggestions = []
    if "疲惫" in [k["label"] for k in keyword_stats[:2]]:
        suggestions.append("这周'累'出现得最多。可以先只做一件事：把睡觉时间往前挪半小时。")
    if "学业" in [k["label"] for k in keyword_stats[:2]]:
        suggestions.append("学业压力占了很大比重。试着把最卡的一步写成一个两小时内能完成的动作。")
    if l2plus:
        suggestions.append("这周有几次情绪明显下沉。要不要把这几次的相同点找出来？")
    if not suggestions:
        suggestions.append("继续保持记录。曲线长一点，才看得清变化。")

    # 逐日情绪分：让周报能指出"哪天最低"，而不是回避说不清楚
    _buckets: dict = {}
    for r in rows:
        d = dt.datetime.fromtimestamp(r["created_at"]).date()
        s = int(round(max(2, min(98, 50 + (r["valence"] or 0) * 45))))
        _buckets.setdefault(d, []).append(s)
    daily = [{"date": d.strftime("%m-%d"), "avg": round(sum(v) / len(v), 1)}
             for d, v in sorted(_buckets.items())]

    return {
        "daily": daily,
        "range": [start.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")],
        "days": len({dt.datetime.fromtimestamp(r["created_at"]).date() for r in rows}),
        "total": len(rows),
        "avg": avg,
        "headline": headline,
        "topics": keyword_stats[:4],
        "distribution": _emotion_dist(rows),
        "suggestions": suggestions,
        "l2plus": l2plus,
    }


def _emotion_dist(rows) -> list[dict]:
    from engine import EMOTION_CN
    counter: dict[str, int] = {}
    for r in rows:
        e = r["emotion"] or "sadness"
        counter[e] = counter.get(e, 0) + 1
    total = sum(counter.values()) or 1
    return [{"emotion": e, "label": EMOTION_CN.get(e, e), "count": c,
             "pct": round(c * 100 / total)} for e, c in
            sorted(counter.items(), key=lambda kv: -kv[1])]


# ---------------------------------------------------------------------------
# 校园聚合指标：不含任何个体数据
# ---------------------------------------------------------------------------

def campus_index(conn: sqlite3.Connection, days: int = 7) -> dict:
    since = int((dt.datetime.now() - dt.timedelta(days=days)).timestamp())
    row = conn.execute(
        """SELECT COUNT(*) n, AVG(valence) v,
                  SUM(CASE WHEN risk_level IN ('L2','L3') THEN 1 ELSE 0 END) hi,
                  COUNT(DISTINCT user_id) u
           FROM messages WHERE role='user' AND created_at>=?""", (since,)).fetchone()
    n = row["n"] or 0
    MIN_SAMPLE = 20
    if n < MIN_SAMPLE:
        return {"available": False, "reason": f"近 {days} 天样本仅 {n} 条，未达 {MIN_SAMPLE} 条最小统计口径，"
                                              "不输出指数（避免个体被反推）", "n": n}
    avg_v = row["v"] or 0
    pressure = int(round(max(0, min(100, 50 - avg_v * 50))))
    return {
        "available": True,
        "days": days,
        "samples": n,
        "active_users": row["u"],
        "pressure_index": pressure,
        "high_ratio": round((row["hi"] or 0) * 100 / n, 1),
        "note": "仅展示整体趋势，不含任何可定位到个人的字段；样本 <20 条时不输出。",
    }
