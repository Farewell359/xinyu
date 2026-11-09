/* ===========================================================
   心屿 XINYU · 前端逻辑
   =========================================================== */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const state = {
  uid: localStorage.getItem("xinyu_uid") || "",
  consent: false,
  busy: false,
  lastAnalysis: null,
  hotlines: [],
  model: null,
  providers: {},
};

const EMOTION_COLOR = {
  sadness: "#5B7C9D",
  joy: "#D8A24A",
  love: "#C98C7A",
  anger: "#C25B4E",
  fear: "#8C7CA8",
  surprise: "#8FAF9D",
};

const RISK_TEXT = { L0: "平稳", L1: "关注", L2: "需要关心", L3: "高风险" };

/* ---------------- 基础请求 ---------------- */
async function api(path, { method = "GET", body } = {}) {
  const url = path.startsWith("http") ? path : path;
  const opt = { method, headers: {} };
  if (body) {
    opt.headers["Content-Type"] = "application/json";
    opt.body = JSON.stringify(body);
  }
  const res = await fetch(url, opt);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `请求失败 (${res.status})`);
  }
  return res.json();
}

function toast(text, ms = 2200) {
  const el = $("#toast");
  el.textContent = text;
  el.classList.add("show");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove("show"), ms);
}

function fmtTime(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
}
function fmtDate(ts) {
  const d = new Date(ts * 1000);
  return `${d.getMonth() + 1}月${d.getDate()}日 ${fmtTime(ts)}`;
}

/* ---------------- 启动 ---------------- */
async function boot() {
  tickClock();
  setInterval(tickClock, 30000);

  try {
    const s = await api("/api/session", { method: "POST", body: { uid: state.uid || null } });
    state.uid = s.uid;
    state.consent = s.consent;
    localStorage.setItem("xinyu_uid", s.uid);
  } catch (e) {
    toast("服务未启动，请先运行 python app.py");
    return;
  }

  const hl = await api("/api/hotlines");
  state.hotlines = hl.hotlines;

  await refreshModelStatus();
  await loadHistory();
  await loadProfile();
  await loadAlerts();
  await loadSettings();
  checkOpeningSheet();
}

/* ---------------- AI 状态 ---------------- */
async function refreshModelStatus() {
  try {
    const data = await api("/api/model");
    state.model = data.config;
    state.providers = data.providers;
    renderAiStatus();
    return data;
  } catch (e) {
    return null;
  }
}

function renderAiStatus() {
  const el = $("#aiStatus");
  const txt = $("#aiStatusText");
  const m = state.model;
  el.classList.remove("on", "off");
  if (m && m.ready) {
    el.classList.add("on");
    txt.textContent = m.provider_name;
    el.title = `AI 已启用：${m.provider_name} / ${m.model}\n点击可管理模型`;
  } else {
    el.classList.add("off");
    txt.textContent = "AI 未连接";
    el.title = "点此填入 API Key，启用真正的 AI 对话";
  }
}

function tickClock() {
  const d = new Date();
  $("#clock").textContent =
    `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
}

/* 首次进入的知情同意（对应"内测前签署知情同意"） */
function checkOpeningSheet() {
  if (!state.consent) {
    openSheet("aboutSheet");
    const btn = $("#aboutSheet .btn.primary");
    btn.textContent = "我已了解，开始使用";
    btn.onclick = async () => {
      await api("/api/consent", { method: "POST", body: { uid: state.uid, consent: true } });
      state.consent = true;
      $("#aboutSheet").classList.remove("open");
      toast("已开启。你的记录只在这台设备上。");
    };
  }
}

/* ---------------- 导航 ---------------- */
const TITLES = {
  chat: ["匿名树洞", "不评判 · 不传出去 · 无需真名"],
  profile: ["情绪档案", "一条曲线，比任何一次对话都诚实"],
  warm: ["暖光预警", "只在需要的时候，轻轻敲一下门"],
  setting: ["我的", "匿名身份 · 引擎状态 · 数据边界"],
};

function switchView(name) {
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === name));
  $("#pageTitle").textContent = TITLES[name][0];
  $("#pageSub").textContent = TITLES[name][1];
  if (name === "profile") loadProfile();
  if (name === "warm") loadAlerts();
  if (name === "setting") loadSettings();
}
$$(".tab").forEach((t) => t.addEventListener("click", () => switchView(t.dataset.view)));
$("#btnInfo").addEventListener("click", () => openSheet("aboutSheet"));

function openSheet(id) { $("#" + id).classList.add("open"); }
$$(".sheet-mask").forEach((m) => {
  m.addEventListener("click", (e) => { if (e.target === m) m.classList.remove("open"); });
});

/* ---------------- 对话 ---------------- */
async function loadHistory() {
  const data = await api(`/api/history?uid=${state.uid}`);
  const box = $("#messages");
  box.innerHTML = "";
  $("#opening").style.display = data.messages.length ? "none" : "";
  data.messages.forEach((m) => renderMessage(m));
  scrollBottom();
}

function renderMessage(m) {
  const isUser = m.role === "user";
  const wrap = document.createElement("div");
  wrap.className = `msg ${isUser ? "me" : "ai"}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const text = document.createElement("div");
  text.textContent = m.content;
  bubble.appendChild(text);

  const meta = document.createElement("div");
  meta.className = "bubble-meta";

  if (isUser && m.emotion) {
    const t = document.createElement("span");
    t.className = "tag";
    t.textContent = emoLabel(m.emotion);
    t.style.color = EMOTION_COLOR[m.emotion] || "";
    meta.appendChild(t);
    if (m.risk_level && m.risk_level !== "L0") {
      const r = document.createElement("span");
      r.className = `tag risk-${m.risk_level}`;
      r.textContent = `${RISK_TEXT[m.risk_level]}`;
      meta.appendChild(r);
    }
  }
  if (!isUser) {
    // 依据可查
    if (m.sources && m.sources.length && m.sources[0].id !== "fallback") {
      const btn = document.createElement("button");
      btn.className = "src-btn";
      btn.textContent = `依据 ${m.sources.length} 条`;
      const list = document.createElement("div");
      list.className = "src-list";
      list.innerHTML = m.sources
        .map((s) => `<p><b>${esc(s.topic)}</b><br/>${esc(s.content)}</p>`)
        .join("");
      btn.onclick = () => list.classList.toggle("show");
      meta.appendChild(btn);
      bubble.appendChild(meta);
      bubble.appendChild(list);
      wrap.appendChild(bubble);
      return wrap;
    }
    if (m.risk_level === "L3") {
      bubble.classList.add("crisis");
      const t = document.createElement("span");
      t.className = "tag";
      t.textContent = "危机响应协议";
      meta.appendChild(t);
    }
  }

  if (meta.childNodes.length) bubble.appendChild(meta);
  wrap.appendChild(bubble);
  $("#messages").appendChild(wrap);
  return wrap;
}

function emoLabel(key) {
  return { sadness: "低落", joy: "愉悦", love: "温暖", anger: "烦躁", fear: "焦虑", surprise: "惊讶" }[key] || key;
}

function esc(s) {
  return String(s || "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function scrollBottom() {
  const el = $("#chatScroll");
  el.scrollTop = el.scrollHeight;
}

function addTyping() {
  const wrap = document.createElement("div");
  wrap.className = "msg ai";
  wrap.id = "typing";
  wrap.innerHTML = `<div class="bubble typing"><i></i><i></i><i></i></div>`;
  $("#messages").appendChild(wrap);
  scrollBottom();
  return wrap;
}

async function send(text) {
  text = (text || $("#input").value).trim();
  if (!text || state.busy) return;
  state.busy = true;
  $("#btnSend").disabled = true;
  $("#input").value = "";
  $("#input").style.height = "auto";
  $("#opening").style.display = "none";

  renderMessage({ role: "user", content: text });
  scrollBottom();
  const typing = addTyping();

  const t0 = performance.now();
  let meta = null;
  let node = null;
  let bubble = null;
  let replyText = "";
  let finished = false;

  // 保险丝：万一底层读流因任何未知原因永不 resolve，
  // 3 分钟后强制解锁，避免再次出现「刷新后只能发一条」。
  const hardTimeout = setTimeout(() => {
    if (!state.busy) return;
    state.busy = false;
    $("#btnSend").disabled = false;
    toast("这次响应超时了，已经解锁，可以重新发送");
  }, 180000);

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ uid: state.uid, text }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `请求失败 (${res.status})`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    // 收到 done / end 就主动收流，绝不等连接自己断——
    // 否则一旦服务端没关连接，reader.read() 会永久挂起，
    // state.busy 卡在 true，按钮一直 disabled（「只能发一次」的根因之一）。
    let streamClosed = false;

    while (!streamClosed) {
      let chunk;
      try { chunk = await reader.read(); } catch { break; }
      if (!chunk || chunk.done) break;
      buf += decoder.decode(chunk.value, { stream: true });

      // 按 SSE 事件切分
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const raw = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const evMatch = raw.match(/^event:\s*(.+)$/m);
        const dataMatch = raw.match(/^data:\s*(.+)$/m);
        if (!dataMatch) continue;
        const ev = evMatch ? evMatch[1].trim() : "message";
        let payload = {};
        try { payload = JSON.parse(dataMatch[1]); } catch { continue; }

        if (ev === "meta") {
          meta = payload;
          state.lastAnalysis = payload;
          typing.remove();
          ({ node, bubble } = createAiBubble(meta));
          bubble.classList.add("streaming");
          updateInspector(meta);
          scrollBottom();
        } else if (ev === "delta" || ev === "replace") {
          if (!bubble) continue;
          replyText = ev === "replace" ? payload.text : replyText + payload.text;
          bubble.firstChild.textContent = replyText;
          scrollBottom();
        } else if (ev === "fallback") {
          if (!bubble) continue;
          replyText = payload.text;
          bubble.firstChild.textContent = replyText;
          meta.ai_error = payload.error;
          meta.mode = "local";
          scrollBottom();
        } else if (ev === "done") {
          finished = true;
          streamClosed = true;
          replyText = payload.text || replyText;
          const elapsed = Math.round(performance.now() - t0);
          if (bubble) {
            bubble.firstChild.textContent = replyText;
            bubble.classList.remove("streaming");
            finalizeAiBubble(node, meta, payload, replyText, elapsed);
          }
          if (payload.alert) {
            setTimeout(() => showWarmSheet(payload.alert, meta), 620);
          }
          if (meta) {
            meta.latency_ms = payload.latency_ms || elapsed;
            updateInspector(meta);
          }
        } else if (ev === "end") {
          // 服务端收尾事件，收到即结束本轮，不再等连接关闭
          streamClosed = true;
        }
      }
    }

    try { await reader.cancel(); } catch {}
    if (!finished) throw new Error("连接中断，请重试");
    await loadAlerts();
  } catch (e) {
    typing.remove();
    if (bubble) bubble.classList.remove("streaming");
    toast(e.message || "发送失败");
  } finally {
    clearTimeout(hardTimeout);
    state.busy = false;
    $("#btnSend").disabled = false;
    $("#input").focus();
  }
}

/* 创建 AI 气泡骨架（流式写入用） */
function createAiBubble(meta) {
  const wrap = document.createElement("div");
  wrap.className = "msg ai";
  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const text = document.createElement("div");
  text.textContent = "";
  bubble.appendChild(text);

  // AI 洞察：让"模型真的理解了"看得见
  const ins = meta.insight || {};
  const hasInsight = ins.core_need || (ins.hidden && ins.hidden !== "无明显潜台词");
  if (hasInsight) {
    const box = document.createElement("div");
    box.className = "ai-insight";
    const parts = [];
    if (ins.core_need) parts.push(`<b>我理解到你想要的是</b>：${esc(ins.core_need)}`);
    if (ins.hidden && ins.hidden !== "无明显潜台词") parts.push(`<b>没说出口的</b>：${esc(ins.hidden)}`);
    if (ins.cognitive && ins.cognitive !== "未见明显") parts.push(`<b>注意到</b>：你可能正处在「${esc(ins.cognitive)}」里`);
    box.innerHTML = parts.join("<br/>");
    bubble.appendChild(box);
  }

  wrap.appendChild(bubble);
  $("#messages").appendChild(wrap);
  scrollBottom();
  return { node: wrap, bubble };
}

/* 收尾：补上元信息（依据 / 来源 / 引擎对照 / 纠正入口） */
function finalizeAiBubble(node, meta, done, replyText, elapsed) {
  if (!node) return;
  const bubble = node.querySelector(".bubble");

  if (meta && meta.risk && meta.risk.level === "L3") bubble.classList.add("crisis");

  const foot = document.createElement("div");
  foot.className = "bubble-meta";

  if (meta && meta.mode === "safety") {
    const t = document.createElement("span");
    t.className = "tag";
    t.textContent = "危机响应协议";
    foot.appendChild(t);
  } else if (meta && meta.mode === "ai") {
    const t = document.createElement("span");
    t.className = "tag";
    t.textContent = "AI 生成";
    foot.appendChild(t);
  } else {
    const t = document.createElement("span");
    t.className = "tag";
    t.textContent = "本地引擎";
    foot.appendChild(t);
  }

  // 引擎对照
  if (meta && meta.engines) {
    const eg = document.createElement("div");
    eg.className = "engines";
    const r = meta.engines.rule;
    eg.innerHTML = `<em>规则 ${esc(r.emotion)}·${r.risk}</em>`;
    if (meta.engines.ai) {
      const a = meta.engines.ai;
      eg.innerHTML += `<em class="ai">AI ${esc(a.emotion)}·${a.risk}</em>`;
    }
    bubble.appendChild(eg);
  }

  // 依据
  if (meta && meta.sources && meta.sources.length && meta.sources[0].id !== "fallback") {
    const btn = document.createElement("button");
    btn.className = "src-btn";
    btn.textContent = `依据 ${meta.sources.length} 条`;
    const list = document.createElement("div");
    list.className = "src-list";
    list.innerHTML = meta.sources
      .map((s) => `<p><b>${esc(s.topic)}</b><br/>${esc(s.content)}</p>`)
      .join("");
    btn.onclick = () => list.classList.toggle("show");
    foot.appendChild(btn);
    bubble.appendChild(foot);
    bubble.appendChild(list);
  } else {
    bubble.appendChild(foot);
  }

  // 异常提示：不装作没事
  if (meta && meta.ai_error) {
    const err = document.createElement("div");
    err.className = "bubble-meta";
    err.innerHTML = `<span class="tag risk-L2">${esc(meta.ai_error)}</span>`;
    bubble.appendChild(err);
  }

  // 误判纠正
  if (meta && meta.emotion) {
    const fix = document.createElement("button");
    fix.className = "fix-btn";
    fix.textContent = "判断不准？";
    fix.onclick = () => askCorrect(0, meta.emotion.label);
    foot.appendChild(fix);
  }
}


$("#btnSend").addEventListener("click", () => send());
$("#input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
$("#input").addEventListener("input", (e) => {
  e.target.style.height = "auto";
  e.target.style.height = Math.min(e.target.scrollHeight, 110) + "px";
});
$$("#chips button").forEach((b) => b.addEventListener("click", () => send(b.dataset.t)));

/* 纠正误判 */
async function askCorrect(messageId, judged) {
  const opts = ["sadness", "joy", "love", "anger", "fear", "surprise"];
  const body = $("#warmSheetBody");
  body.innerHTML = `
    <div class="sheet-grab"></div>
    <p class="sheet-kicker">帮我改准一点</p>
    <h2 class="sheet-title">这句话，其实更接近哪种心情？</h2>
    <p class="sheet-text">当前判断：<b>${emoLabel(judged)}</b>。你的纠正会回流到评测集，用来调整下一轮的阈值 —— 我们允许自己被纠正。</p>
    <div class="sheet-actions" id="correctOpts">
      ${opts.map((o) => `<button data-v="${o}">${emoLabel(o)}</button>`).join("")}
      <button class="primary" data-v="skip">其实判断得挺准</button>
    </div>`;
  openSheet("warmSheet");
  $$("#correctOpts button").forEach((b) =>
    b.addEventListener("click", async () => {
      const v = b.dataset.v;
      if (v !== "skip") {
        await api("/api/feedback", {
          method: "POST",
          body: { uid: state.uid, message_id: messageId, judged, corrected: v },
        });
        toast("谢谢你，已记录");
      }
      $("#warmSheet").classList.remove("open");
      loadSettings();
    })
  );
}

/* ---------------- 暖光弹层 ---------------- */
function showWarmSheet(alert, r) {
  const isL3 = alert.level === "L3";
  const msg = alert.message || {};
  const body = $("#warmSheetBody");
  body.innerHTML = `
    <div class="sheet-grab"></div>
    <p class="sheet-kicker" style="color:${isL3 ? "#C25B4E" : "#D8A24A"}">${isL3 ? "请先停下来看这一页" : "暖光提醒 · 只发给你"}</p>
    <h2 class="sheet-title">${esc(msg.title)}</h2>
    <p class="sheet-text">${esc(msg.body)}</p>
    ${isL3 ? `<div class="alert-item L3" style="margin:12px 0">
        <b>如果此刻有具体的念头：</b><br/>
        请先拨打 <b>12356</b>（全国心理援助热线，24 小时），或者立刻找一个身边的人待在一起。
        你不需要说清楚发生了什么，只要说"我现在需要人陪"就够了。
      </div>` : ""}
    <div class="sheet-actions">
      ${state.hotlines
        .slice(0, 3)
        .map((h) => `<button data-tel="${h.number}" ${/[0-9]{5,}/.test(h.number) ? "" : "disabled"}>
            ${esc(h.name)} · ${esc(h.number)}</button>`)
        .join("")}
      ${(msg.cta || []).map((c, i) => `<button class="${i === 0 ? "primary" : ""}" data-cta="${i}">${esc(c)}</button>`).join("")}
    </div>`;
  openSheet("warmSheet");
  $$("#warmSheetBody button[data-tel]").forEach((b) =>
    b.addEventListener("click", () => {
      const n = b.dataset.tel;
      if (/^[0-9-]+$/.test(n)) window.location.href = "tel:" + n.replace(/-/g, "");
      else toast("可拨打 12356 —— 全国 24 小时心理援助热线");
    })
  );
  $$("#warmSheetBody button[data-cta]").forEach((b) =>
    b.addEventListener("click", () => {
      if (b.dataset.cta === "0" && !isL3) { $("#warmSheet").classList.remove("open"); switchView("warm"); }
      else { $("#warmSheet").classList.remove("open"); }
    })
  );
}

/* ---------------- 情绪档案 ---------------- */
async function loadProfile() {
  const p = await api(`/api/profile?uid=${state.uid}&days=30`);
  const s = p.summary;
  $("#pAvg").textContent = s.avg == null ? "—" : Math.round(s.avg);
  $("#pCount").textContent = s.total;
  $("#pStreak").textContent = s.streak;
  $("#pBest").textContent = s.best == null ? "—" : s.best;

  const tp = $("#pTrend");
  const trendMap = {
    up: ["up", "在回升"], down: ["down", "在走低"], flat: ["", "大致平稳"], insufficient: ["", "还需要更多记录"],
  };
  tp.className = "trend-pill " + (trendMap[s.trend]?.[0] || "");
  tp.textContent = trendMap[s.trend]?.[1] || "—";

  drawSparkline(p.points);

  const w = await api(`/api/weekly?uid=${state.uid}`);
  const bars = $("#distBars");
  if (!w.topics.length) {
    bars.innerHTML = `<p class="empty-tip">这周还没有足够的关键词。多说几句，这里会慢慢长出形状。</p>`;
  } else {
    const max = Math.max(...w.topics.map((t) => t.count));
    bars.innerHTML = w.topics
      .map((t) => `<div class="bar"><span>${esc(t.label)}</span>
        <span class="track"><i class="fill" style="width:0" data-w="${(t.count / max) * 100}"></i></span>
        <b>${t.count}</b></div>`)
      .join("");
    requestAnimationFrame(() => $$(".bar .fill").forEach((f) => (f.style.width = f.dataset.w + "%")));
  }

  const mix = $("#emotionMix");
  const all = await api(`/api/profile?uid=${state.uid}&days=30`);
  const counts = {};
  all.points.forEach((p) => (counts[p.emotion] = (counts[p.emotion] || 0) + 1));
  const total = all.points.length || 1;
  const keys = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
  mix.innerHTML = keys.length
    ? keys
        .map(
          (k) => `<div><i style="background:${EMOTION_COLOR[k] || "#999"}"></i>${emoLabel(k)}
        <b>${Math.round((counts[k] / total) * 100)}%</b></div>`
        )
        .join("")
    : `<p class="empty-tip">还没有记录。</p>`;
}

function drawSparkline(points) {
  const svg = $("#sparkline");
  const W = 320, H = 120, PAD = 8;
  if (!points.length) {
    svg.innerHTML = `<text x="160" y="66" text-anchor="middle" font-size="11" fill="#AEB8B1">还没有数据</text>`;
    return;
  }
  // 按时间映射到 30 天窗口
  const now = Date.now() / 1000;
  const span = 30 * 86400;
  const t0 = now - span;
  const xs = points.map((p) => PAD + ((p.ts - t0) / span) * (W - PAD * 2));
  const ys = points.map((p) => H - PAD - (p.score / 100) * (H - PAD * 2));

  let path = "";
  let area = "";
  if (points.length === 1) {
    path = `M${xs[0]},${ys[0]} l0.1,0`;
    area = "";
  } else {
    // 平滑曲线
    for (let i = 0; i < xs.length; i++) {
      if (i === 0) { path += `M${xs[0]},${ys[0]}`; continue; }
      const x0 = xs[i - 1], y0 = ys[i - 1], x1 = xs[i], y1 = ys[i];
      const cx = (x0 + x1) / 2;
      path += ` C${cx},${y0} ${cx},${y1} ${x1},${y1}`;
    }
    area = `${path} L${xs[xs.length - 1]},${H - PAD} L${xs[0]},${H - PAD} Z`;
  }

  const dots = points
    .map(
      (p, i) =>
        `<circle cx="${xs[i]}" cy="${ys[i]}" r="${i === points.length - 1 ? 4 : 2.6}"
           fill="${EMOTION_COLOR[p.emotion] || "#fff"}" stroke="#2F5D50" stroke-width="1.2"/>`
    )
    .join("");

  // 50 分参考线
  const midY = H - PAD - 0.5 * (H - PAD * 2);

  svg.innerHTML = `
    <defs>
      <linearGradient id="gArea" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#F0D49A" stop-opacity=".38"/>
        <stop offset="100%" stop-color="#F0D49A" stop-opacity="0"/>
      </linearGradient>
    </defs>
    <line x1="${PAD}" y1="${midY}" x2="${W - PAD}" y2="${midY}" stroke="rgba(255,255,255,.16)" stroke-dasharray="3 4"/>
    ${area ? `<path d="${area}" fill="url(#gArea)"/>` : ""}
    <path d="${path}" fill="none" stroke="#F0D49A" stroke-width="2" stroke-linecap="round"/>
    ${dots}`;
}

/* 周报 */
$("#btnWeekly").addEventListener("click", async () => {
  const body = $("#weeklyBody");
  body.innerHTML = `<div class="sheet-grab"></div><p class="sheet-kicker">每周小结</p>
    <p class="wk-headline" style="margin-top:14px;color:#8A9891">正在为你写这一周的小结…</p>`;
  openSheet("weeklySheet");

  let w;
  try {
    w = await api(`/api/weekly?uid=${state.uid}`);
  } catch (e) {
    body.innerHTML = `<div class="sheet-grab"></div><p class="sheet-kicker">每周小结</p>
      <p class="wk-headline" style="color:#C25B4E">加载失败：${esc(e.message)}</p>`;
    return;
  }

  body.innerHTML = `
    <div class="sheet-grab"></div>
    <p class="sheet-kicker">每周小结${w.narrative_source === "ai" ? " · AI 撰写" : ""}</p>
    <p class="wk-range">${w.range[0]} — ${w.range[1]} · 记录 ${w.days} 天</p>
    <h2 class="sheet-title" style="margin-top:10px">这一周的你</h2>
    ${w.narrative
      ? `<div class="wk-narrative">${esc(w.narrative).replace(/\n+/g, "</p><p>").replace(/^/, "<p>")}</div>`
      : `<p class="wk-headline">${esc(w.headline)}</p>`}
    <div class="wk-grid">
      <div><b>${w.avg == null ? "—" : Math.round(w.avg)}</b><span>平均分</span></div>
      <div><b>${w.total}</b><span>次倾诉</span></div>
      <div><b>${w.l2plus}</b><span>次明显下沉</span></div>
    </div>
    ${w.topics.length ? `<div class="wk-sec"><h4>常出现的话题</h4>
      ${w.topics.map((t) => `<p>${esc(t.label)} · 提到 ${t.count} 次</p>`).join("")}</div>` : ""}
    ${w.narrative ? "" : `<div class="wk-sec"><h4>可以试试</h4>
      ${w.suggestions.map((s) => `<p>${esc(s)}</p>`).join("")}</div>`}
    <p class="sheet-foot">${w.narrative ? "由 AI 根据你的真实统计数据撰写。" : ""}这份小结只有你能看到，也不会被用于任何评估。</p>
    <button class="btn primary" onclick="document.getElementById('weeklySheet').classList.remove('open')">收好</button>`;
});

/* ---------------- 暖光页 ---------------- */
async function loadAlerts() {
  const { alerts } = await api(`/api/alerts?uid=${state.uid}`);
  const list = $("#alertList");
  $("#alertCount").textContent = `${alerts.length} 条`;
  list.innerHTML = alerts.length
    ? alerts
        .map(
          (a) => `<div class="alert-item ${a.level}">
        <b>${RISK_TEXT[a.level]} · ${a.level}</b><br/>${esc(a.reason || "")}
        <time>${fmtDate(a.created_at)}</time></div>`
        )
        .join("")
    : `<p class="empty-tip">目前没有需要提醒的事。这不是异常，是这段时间你还撑得挺好。</p>`;

  const hl = $("#hotlineList");
  hl.innerHTML = state.hotlines
    .map(
      (h) => `<div class="hotline"><div>
        <div class="name">${esc(h.name)}</div>
        <div class="note">${esc(h.note)}</div></div>
        <a href="${/^[0-9-]+$/.test(h.number) ? "tel:" + h.number.replace(/-/g, "") : "#"}"
           ${/^[0-9-]+$/.test(h.number) ? "" : 'style="background:#B9C2BB"'}>${esc(h.number)}</a>
      </div>`
    )
    .join("");
}

/* ---------------- 设置 ---------------- */
async function loadSettings() {
  $("#uidText").textContent = state.uid;
  const kb = await api("/api/kb");
  $("#kbCount").textContent = `${kb.count} 条`;

  await refreshModelStatus();
  const m = state.model;
  $("#engineMode").textContent = m && m.ready ? "AI 已启用" : "本地引擎";
  $("#modelState").textContent = m && m.ready ? "已启用" : "未配置";
  $("#modelState").style.color = m && m.ready ? "#4CAF7D" : "#C25B4E";
  $("#engineAI").textContent = m && m.ready ? `${m.provider_name} / ${m.model}` : "未启用（规则引擎）";

  fillModelForm(m);

  const c = await api("/api/campus?days=7");
  const box = $("#campusBox");
  if (c.available) {
    box.innerHTML = `
      <div class="big"><b>${c.pressure_index}</b><span>/ 100 本周求助压力指数</span></div>
      <div class="kv"><span>样本条数</span><b>${c.samples}</b></div>
      <div class="kv"><span>活跃匿名用户</span><b>${c.active_users}</b></div>
      <div class="kv"><span>需关心比例</span><b>${c.high_ratio}%</b></div>
      <p>${esc(c.note)}</p>`;
  } else {
    box.innerHTML = `<p class="empty-tip">${esc(c.reason)}</p>
      <p class="tiny">设计如此：数据太少时宁可不出，也不让任何一个人被认出来。</p>`;
  }
}

/* ---------------- AI 模型配置 ---------------- */
function fillModelForm(m) {
  const sel = $("#mProvider");
  if (!sel.options.length) {
    sel.innerHTML = Object.entries(state.providers)
      .map(([k, v]) => `<option value="${k}">${esc(v.name)}</option>`)
      .join("");
    sel.addEventListener("change", () => {
      const p = state.providers[sel.value];
      if (!p) return;
      $("#mBase").value = p.base_url;
      $("#mModel").value = p.model;
      renderProviderNote(sel.value);
    });
  }
  if (m) {
    sel.value = m.provider || "deepseek";
    $("#mBase").value = m.base_url || "";
    $("#mModel").value = m.model || "";
    $("#mKey").value = "";
    $("#mKey").placeholder = m.has_key ? `已保存 ${m.masked_key}（留空则不修改）` : "粘贴你的 API Key";
  }
  renderProviderNote(sel.value);
}

function renderProviderNote(provider) {
  const p = state.providers[provider];
  if (!p) return;
  const note = $("#mNote");
  note.innerHTML = `${esc(p.note)}` + (p.apply ? ` · <a href="${p.apply}" target="_blank" rel="noopener">获取 Key</a>` : "");
}

$("#btnTest").addEventListener("click", async () => {
  const btn = $("#btnTest");
  const out = $("#mResult");
  btn.disabled = true;
  btn.textContent = "测试中…";
  out.className = "model-result";
  try {
    const payload = {
      provider: $("#mProvider").value,
      base_url: $("#mBase").value.trim(),
      model: $("#mModel").value.trim(),
    };
    const k = $("#mKey").value.trim();
    if (k) payload.api_key = k;
    const r = await api("/api/model/test", { method: "POST", body: payload });
    $("#mTried").innerHTML = "";
    if (r.ok) {
      out.className = "model-result show ok";
      out.textContent = `✓ 连接成功（${r.latency_ms} ms）模型 ${r.model} 已可用，回复："${r.reply}"`;
    } else {
      out.className = "model-result show err";
      out.textContent = `✗ ${r.error}${r.kind === "ratelimit" ? "（可重试）" : ""}\n${r.detail || ""}`;
      // 未开通 / 模型名不对 → 直接把「探测可用模型」推给用户
      if (r.kind === "nomodel") {
        out.textContent += "\n\n👉 点右边「探测可用模型」，我挨个试一遍告诉你哪个能用。";
      }
    }
  } catch (e) {
    out.className = "model-result show err";
    out.textContent = `✗ ${e.message}`;
  } finally {
    btn.disabled = false;
    btn.textContent = "测试连接";
  }
});

$("#btnProbe").addEventListener("click", async () => {
  const btn = $("#btnProbe");
  const out = $("#mResult");
  const tried = $("#mTried");
  btn.disabled = true;
  btn.textContent = "挨个试…";
  out.className = "model-result show";
  out.textContent = "正在依次测试该服务商的候选模型（每试一个约 1-3 秒）…";
  tried.innerHTML = "";
  try {
    const payload = {
      provider: $("#mProvider").value,
      base_url: $("#mBase").value.trim(),
    };
    const k = $("#mKey").value.trim();
    if (k) payload.api_key = k;
    const r = await api("/api/model/probe", { method: "POST", body: payload });

    // 逐条列出结果，用户能一眼看到「哪个能用、哪个没开通、哪个被限流」
    const label = { nomodel: "未开通", ratelimit: "被限流", noauth: "Key 无效",
                    balance: "欠费", server: "服务商故障", net: "网络问题", other: "失败" };
    tried.innerHTML = (r.tried || []).map((t) => `
      <div class="tried-row ${t.ok ? "ok" : "err"}">
        <span class="tried-name">${esc(t.model)}</span>
        <span class="tried-tag">${t.ok ? `可用 ${t.latency_ms || ""}ms` : (label[t.kind] || "失败")}</span>
      </div>`).join("");

    if (r.ok) {
      out.className = "model-result show ok";
      out.textContent = `✓ ${r.message}\n已自动填入模型名，记得点「保存并启用」才会生效。`;
      $("#mModel").value = r.model;
    } else {
      out.className = "model-result show err";
      out.textContent = `✗ ${r.error}\n${r.detail || ""}`;
    }
  } catch (e) {
    out.className = "model-result show err";
    out.textContent = `✗ ${e.message}`;
  } finally {
    btn.disabled = false;
    btn.textContent = "探测可用模型";
  }
});

$("#btnSaveModel").addEventListener("click", async () => {
  const btn = $("#btnSaveModel");
  const out = $("#mResult");
  btn.disabled = true;
  btn.textContent = "保存中…";
  try {
    const payload = {
      provider: $("#mProvider").value,
      base_url: $("#mBase").value.trim(),
      model: $("#mModel").value.trim(),
    };
    const k = $("#mKey").value.trim();
    if (k) payload.api_key = k;
    const r = await api("/api/model", { method: "POST", body: payload });
    state.model = r.config;
    renderAiStatus();
    await loadSettings();
    out.className = "model-result show ok";
    out.textContent = r.config.ready
      ? `✓ 已启用 ${r.config.provider_name} / ${r.config.model}。回到树洞说一句话，回复将由 AI 生成。`
      : "已保存，但还缺少必填项（Key / 模型名）";
    if (r.config.ready) toast("AI 已启用，去树洞试试");
  } catch (e) {
    out.className = "model-result show err";
    out.textContent = `✗ ${e.message}`;
  } finally {
    btn.disabled = false;
    btn.textContent = "保存并启用";
  }
});

$("#aiStatus").addEventListener("click", () => {
  switchView("setting");
  setTimeout(() => $("#mKey").focus(), 260);
});

$("#btnResetId").addEventListener("click", () => {
  localStorage.removeItem("xinyu_uid");
  location.reload();
});
$("#btnClear").addEventListener("click", async () => {
  if (!confirm("确定清空这台设备上的所有倾诉记录？清空后无法恢复。")) return;
  await api("/api/reset", { method: "POST", body: { uid: state.uid } });
  toast("已清空");
  await loadHistory();
  await loadProfile();
  await loadAlerts();
  switchView("chat");
});

/* ---------------- 实时流水线面板 ---------------- */
function updateInspector(r) {
  if (!r || !r.emotion) return;
  state.lastAnalysis = r;
  $("#insLatency").textContent = r.latency_ms ? `${r.latency_ms} ms` : "";
  const e = r.emotion;
  const scores = Object.entries(e.scores || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4);
  const bars = scores
    .map(([k, v]) => `<i style="width:${(v * 100).toFixed(1)}%;background:${EMOTION_COLOR[k]}"></i>`)
    .join("");

  const ins = r.insight || {};
  const aiOn = r.mode === "ai";
  const genText = r.mode === "safety"
    ? "危机协议（硬编码，模型无权覆盖）"
    : aiOn ? "AI 生成（知识卡 RAG + 对话上下文）" : "本地规则引擎";

  $("#insBody").innerHTML = `
    <div class="ins-row"><div class="k">脱敏</div><div class="v">${
      (r.sanitize_hits || []).length
        ? `命中 ${r.sanitize_hits.length} 处，已替换为占位符`
        : "未检出身份信息"
    }</div></div>
    <div class="ins-row"><div class="k">清洗后</div><div class="v" style="color:#54655D">${esc(r.sanitized)}</div></div>
    <div class="ins-row"><div class="k">情绪</div><div class="v">
      <b style="color:${EMOTION_COLOR[e.label]}">${e.label_cn}</b>
      <span style="color:#8A9891"> 置信 ${(e.score * 100).toFixed(0)}% · 效价 ${e.valence.toFixed(2)} · 强度 ${(e.intensity * 100).toFixed(0)}%</span>
      <div class="ins-bar">${bars}</div>
      <div class="ins-legend">${scores.map(([k, v]) => `<em>${emoLabel(k)} ${(v * 100).toFixed(0)}%</em>`).join("")}</div>
    </div></div>
    <div class="ins-row"><div class="k">风险</div><div class="v">
      <span class="ins-lv" style="background:${r.risk.meta.color}22;color:${r.risk.meta.color}">${r.risk.level} ${RISK_TEXT[r.risk.level]}</span>
      <span style="color:#8A9891"> 分值 ${r.risk.score}</span>
      <div style="margin-top:5px;color:#54655D;font-size:11px">${esc((r.risk.reasons || [])[0] || "")}</div>
      ${(r.risk.signals || []).length ? `<div style="margin-top:4px;color:#8A9891;font-size:10.5px">信号：${esc(r.risk.signals.slice(0, 4).join("、"))}</div>` : ""}
    </div></div>
    ${ins.core_need || ins.hidden ? `<div class="ins-row"><div class="k">AI 理解</div><div class="v">
      ${ins.core_need ? `<div>核心诉求：<b style="color:#2F5D50">${esc(ins.core_need)}</b></div>` : ""}
      ${ins.hidden && ins.hidden !== "无明显潜台词" ? `<div style="margin-top:4px;color:#54655D">潜台词：${esc(ins.hidden)}</div>` : ""}
      ${ins.cognitive && ins.cognitive !== "未见明显" ? `<div style="margin-top:4px;color:#8A7761">认知模式：${esc(ins.cognitive)}</div>` : ""}
    </div></div>` : ""}
    <div class="ins-row"><div class="k">生成</div><div class="v">
      ${genText}
      ${(r.sources || []).length && r.sources[0].id !== "fallback" ? `<br/><span style="color:#8A9891;font-size:10.5px">引用：${esc(r.sources.map((s) => s.topic).join(" / "))}</span>` : ""}
      ${(r.sources || [])[0] && r.sources[0].id === "fallback" ? `<br/><span style="color:#C25B4E;font-size:10.5px">未命中知识卡 → 仅追问，不建议</span>` : ""}
    </div></div>`;

  // 双引擎对照
  const foot = $("#insFoot");
  if (r.engines) {
    foot.classList.add("show");
    const { rule, ai } = r.engines;
    foot.innerHTML = `规则引擎：${esc(rule.emotion)} · ${rule.risk}　|　` +
      (ai ? `AI 判断：${esc(ai.emotion)} · ${ai.risk}` : "AI 未参与");
  } else {
    foot.classList.remove("show");
  }
}

boot();
