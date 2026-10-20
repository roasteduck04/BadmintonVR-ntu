// BadmintonVR live page (v0.8.2): 1 Calibrate, 2 Stream. Plain JS, no build step.
"use strict";

const $ = (s) => document.querySelector(s);
const state = { schema: null, base: {}, values: {}, clips: [], presets: [], info: null, polling: null, unity: null };

async function api(path, body) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  const t = await r.text();
  let d; try { d = t ? JSON.parse(t) : {}; } catch { d = t; }
  if (!r.ok) { const e = new Error((d && d.detail && (d.detail.message || d.detail)) || r.statusText); e.detail = d && d.detail; e.status = r.status; throw e; }
  return d;
}

function toast(msg, ms = 4000) {
  const t = $("#toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toast.h); toast.h = setTimeout(() => (t.hidden = true), ms);
}

const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(Number(v)) ? "–" : Number(v).toFixed(d));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// ------------------------------------------------------------------ tabs
document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("on", x === b));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("on", t.id === "tab-" + b.dataset.tab));
  if (b.dataset.tab === "calibrate") openCalibrate();
  if (b.dataset.tab === "stream") loadSettings();          // a new calibration / sync file shows in the clip line
}));

// ------------------------------------------------------------------ 1 Calibrate
function openCalibrate() {
  const f = $("#calib-frame");
  if (!f.src) f.src = "/calibrate";
  loadSync();
}

async function loadSync() {
  try {
    const [rows, cams] = await Promise.all([api("/api/sync"), api("/calib/api/cameras")]);
    const tb = $("#sync tbody"); tb.innerHTML = "";
    rows.forEach((r) => {
      const cc = cams.filter((c) => c.clip === r.clip), done = cc.filter((c) => c.calibrated);
      const bad = done.filter((c) => (c.rms || 0) > 15).length;
      const s = r.sync;
      const offs = s ? Object.entries(s.offsets).map(([v, o]) => `${v.replace(/_\d+$/, "")} ${o >= 0 ? "+" : ""}${o}`).join(", ") : "–";
      const tr = document.createElement("tr");
      tr.innerHTML = `<td class="name">${r.clip}</td>
        <td class="${done.length === cc.length ? (bad ? "warnc" : "good") : "badc"}">${done.length} / ${cc.length}${bad ? ` (${bad} over 15 px)` : ""}</td>
        <td>${s ? `${s.made || ""} · from ${s.source || "?"}` : '<span class="muted">none</span>'}</td>
        <td>${offs}</td>
        <td><button class="small" data-sync="${r.clip}" ${r.has_build ? "" : 'disabled title="no offline build of this clip"'}>${s ? "Remake" : "Make"} from the offline build</button></td>`;
      tb.appendChild(tr);
    });
  } catch (e) { toast("sync: " + e.message); }
}

document.addEventListener("click", async (ev) => {
  const c = ev.target.dataset && ev.target.dataset.sync;
  if (!c) return;
  try { const d = await api("/api/sync", { clip: c }); toast(`Sync file for ${c} written${d.archived ? " (old one archived)" : ""}.`); loadSync(); }
  catch (e) { toast(e.message, 7000); }
});

// ------------------------------------------------------------------ settings form
function fieldInput(f, v) {
  const wrap = document.createElement("div"); wrap.className = "in";
  let el;
  if (f.kind === "choice") {
    el = document.createElement("select");
    f.options.forEach(([val, lab]) => { const o = document.createElement("option"); o.value = JSON.stringify(val); o.textContent = lab; el.appendChild(o); });
    el.value = JSON.stringify(v);
    if (el.selectedIndex < 0) { const o = document.createElement("option"); o.value = JSON.stringify(v); o.textContent = String(v); el.appendChild(o); el.value = o.value; }
    el.addEventListener("change", () => setValue(f.key, JSON.parse(el.value)));
    wrap.appendChild(el);
  } else if (f.kind === "range") {
    const a = document.createElement("input"), b = document.createElement("input");
    [a, b].forEach((x) => { x.type = "number"; x.step = f.frames ? "1" : "any"; x.style.width = "90px"; });
    a.value = v && v[0] !== null ? v[0] : ""; b.value = v && v[1] !== null && v[1] !== undefined ? v[1] : "";
    b.placeholder = "end";
    const upd = () => setValue(f.key, [a.value === "" ? 0 : Number(a.value), b.value === "" ? null : Number(b.value)]);
    a.addEventListener("change", upd); b.addEventListener("change", upd);
    wrap.append(a, document.createTextNode("–"), b);
  } else if (f.kind === "optfloat") {
    const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = v !== null && v !== undefined; cb.title = "on / off";
    const n = document.createElement("input"); n.type = "number"; n.step = "any"; n.value = cb.checked ? v : ""; n.disabled = !cb.checked;
    cb.addEventListener("change", () => { n.disabled = !cb.checked; if (cb.checked && n.value === "") n.value = state.base[f.key] ?? 1; setValue(f.key, cb.checked ? Number(n.value) : null); });
    n.addEventListener("change", () => setValue(f.key, Number(n.value)));
    wrap.append(cb, n);
  } else if (f.kind === "bool") {
    el = document.createElement("input"); el.type = "checkbox"; el.checked = !!v;
    el.addEventListener("change", () => setValue(f.key, el.checked)); wrap.appendChild(el);
  } else {
    el = document.createElement("input"); el.type = "number"; el.step = f.kind === "int" ? "1" : "any"; el.value = v ?? "";
    el.addEventListener("change", () => setValue(f.key, el.value === "" ? null : Number(el.value))); wrap.appendChild(el);
  }
  if (f.unit) { const u = document.createElement("span"); u.className = "unit"; u.textContent = f.unit; wrap.appendChild(u); }
  if (f.frames) { const sx = document.createElement("span"); sx.className = "secs"; sx.dataset.secs = f.key; wrap.appendChild(sx); }
  return wrap;
}

// frames fields: the same length in seconds at the chosen output frame rate, greyed next to the input
function updateSecs() {
  const fps = Number(state.values.output_fps) || 0;
  document.querySelectorAll("#groups .secs").forEach((el) => {
    const v = state.values[el.dataset.secs];
    const s = (x) => (x === null || x === undefined || !fps ? null : (x / fps).toFixed(x % fps ? 2 : 1));
    if (Array.isArray(v)) el.textContent = fps ? `= ${s(v[0]) ?? 0} – ${s(v[1]) ?? "end"} s` : "";
    else el.textContent = s(v) !== null ? `= ${s(v)} s` : "";
  });
}

function renderForm() {
  const host = $("#groups"); host.innerHTML = "";
  const live = state.values.backend === "live";
  state.schema.groups.forEach((g) => {
    const fields = state.schema.fields.filter((f) => f.group === g.id);
    const det = document.createElement("details"); det.className = "group"; det.dataset.group = g.id;
    det.open = g.id === "source";
    const off = fields.every((f) => f.live_only) && !live;
    if (off) det.classList.add("disabled");
    det.innerHTML = `<summary>${g.label}<span class="gh">${off ? "only when the 2D comes from the videos" : g.help}</span><span class="badge"></span></summary>`;
    const box = document.createElement("div"); box.className = "fields";
    fields.forEach((f) => {
      const d = document.createElement("div"); d.className = "field"; d.dataset.key = f.key;
      d.innerHTML = `<div class="fl">${f.label}</div>`;
      const inp = fieldInput(f, state.values[f.key]);
      if (f.live_only && !live) inp.querySelectorAll("input,select").forEach((x) => (x.disabled = true));
      d.appendChild(inp);
      const h = document.createElement("div"); h.className = "fh"; h.textContent = f.help; d.appendChild(h);
      box.appendChild(d);
    });
    det.appendChild(box); host.appendChild(det);
  });
  markChanged(); sourceLine();
}

function sourceLine() {
  const v = state.values;
  const src = v.backend === "live" ? "straight from the videos (2D pose on the GPU, frame by frame)" : "the stored 2D files (this machine has no NVIDIA GPU)";
  $("#source-line").innerHTML = `<b>Source:</b> ${src} · ${v.output_fps} fps · ${v.pace === "realtime" ? "real time (camera clock)" : "as fast as possible"}`;
}

function setValue(key, v) {
  const oldFps = Number(state.values.output_fps);
  state.values[key] = v;
  if (key === "output_fps" && oldFps && Array.isArray(state.values.section_s)) {
    // the part of the video stays the same; the other frame counts stay frames (so their seconds change)
    const r = Number(v) / oldFps;
    state.values.section_s = state.values.section_s.map((x) => (x === null ? null : Math.round(x * r)));
    renderForm(); return;
  }
  if (key === "backend") { renderForm(); return; }
  markChanged(); sourceLine();
}

function markChanged() {
  let n = 0;
  document.querySelectorAll("#groups .field").forEach((d) => {
    const c = !same(state.values[d.dataset.key], state.base[d.dataset.key]);
    d.classList.toggle("changed", c); if (c) n++;
  });
  document.querySelectorAll("#groups .group").forEach((g) => {
    const k = g.querySelectorAll(".field.changed").length;
    g.querySelector(".badge").textContent = k ? `${k} changed` : "";
  });
  $("#changed").textContent = n ? `${n} setting${n > 1 ? "s" : ""} changed from the preset` : "";
  updateSecs();
}

async function loadSettings() {
  const clip = $("#clip").value, preset = $("#preset").value;
  if (!clip) return;
  try {
    const d = await api(`/api/settings?clip=${encodeURIComponent(clip)}&preset=${encodeURIComponent(preset)}`);
    state.base = d.values; state.values = JSON.parse(JSON.stringify(d.values));
    renderForm();
  } catch (e) { toast("settings: " + e.message); }
  const p = state.presets.find((x) => x.name === preset);
  $("#preset-about").textContent = p ? p.about : "";
  const c = state.clips.find((x) => x.name === clip);
  if (c) {
    const bits = [`${c.views.length} cameras (${c.views.join(", ")})`, c.section_s ? `section ${c.section_s[0]}–${c.section_s[1] ?? "end"} s` : "",
      c.calibrated ? "calibrated" : "NOT calibrated (tab 1)", c.sync_file ? "sync file present" : "no sync file"];
    $("#clip-info").textContent = bits.filter(Boolean).join(" · ");
  }
}

async function loadPresets(select) {
  state.presets = await api("/api/presets");
  const s = $("#preset"); s.innerHTML = "";
  state.presets.forEach((p) => { const o = document.createElement("option"); o.value = p.name; o.textContent = p.name + (p.builtin ? " (built in)" : ""); s.appendChild(o); });
  s.value = select || state.info.default_settings;
  await loadSettings();
}

$("#clip").addEventListener("change", loadSettings);
$("#preset").addEventListener("change", loadSettings);
$("#reset").addEventListener("click", () => { state.values = JSON.parse(JSON.stringify(state.base)); renderForm(); });
$("#save").addEventListener("click", async () => {
  const name = $("#save-name").value.trim().toLowerCase();
  try {
    await api("/api/presets", { name, values: state.values, about: `saved from the page for ${$("#clip").value}` });
    await loadPresets(name); toast(`Preset ${name} saved (configs/live/settings/${name}.yaml).`);
  } catch (e) { toast(e.message, 6000); }
});

// ------------------------------------------------------------------ start / stop
async function start(force) {
  $("#confirm").hidden = true;
  try {
    $("#start").disabled = true;
    const j = await api("/api/stream/start", { clip: $("#clip").value, preset: $("#preset").value, values: state.values, force: !!force });
    showJob(j);
    toast(state.unity && state.unity.in_play ? "Streaming — Unity shows each pose as its file lands." : "Streaming — Unity can join any time (press Play).", 5000);
    pollJob();
  } catch (e) {
    $("#start").disabled = false;
    if (e.detail && e.detail.code === "unity_not_playing") { $("#confirm-msg").textContent = e.detail.message; $("#confirm").hidden = false; }
    else toast(e.message, 7000);
  }
}
$("#start").addEventListener("click", () => start(false));
$("#confirm-anyway").addEventListener("click", () => start(true));
$("#confirm-cancel").addEventListener("click", () => ($("#confirm").hidden = true));
$("#stop").addEventListener("click", async () => { try { showJob(await api("/api/stream/stop", {})); } catch (e) { toast(e.message); } });

const mmss = (s) => (s == null || !isFinite(s) ? "–" : s < 60 ? `${Math.round(s)} s` : `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`);
const ICON = { pending: "○", done: "✓", failed: "✗", stopped: "■" };

function showJob(j) {
  const box = $("#job");
  box.className = "job " + (j ? j.state : "idle");
  const running = !!(j && (j.state === "running" || j.state === "starting"));
  $("#start").disabled = running; $("#stop").disabled = !running;
  $("#start").textContent = running ? "Streaming…" : "Start streaming to Unity";
  if (!j) {
    $("#job-title").textContent = "Idle"; $("#job-stats").textContent = ""; $("#job-folder").textContent = "";
    $("#phases").innerHTML = ""; $("#bar").className = "bar"; $("#bar-fill").style.width = "0"; $("#bar-text").textContent = "";
    $("#open-folder").hidden = true; return;
  }
  const words = { starting: "Starting", running: "Streaming", done: "Stream finished", failed: "Failed", stopped: "Stopped" };
  $("#job-title").textContent = `${words[j.state] || j.state}: ${j.title}`;
  const p = j.progress || {}, ph = j.phases || [], now = j.now || Date.now() / 1000, meta = j.meta || {};
  // phases: ✓ done (how long it took), spinner = now (how long so far), ○ still to come
  $("#phases").innerHTML = ph.map((x) => {
    const ic = x.state === "active" ? (running ? '<span class="spin"></span>' : "■") : ICON[x.state] || "○";
    const dur = x.t0 ? mmss((x.t1 || (running ? now : j.ended || now)) - x.t0) : "";
    return `<li class="${x.state}"><span class="ic">${ic}</span><span class="pl">${x.label}</span>
      <span class="pd">${x.detail || ""}</span><span class="pt">${dur}</span></li>`;
  }).join("");
  // bar: frames while streaming; moving stripes while loading / saving / writing the report
  const act = ph.find((x) => x.state === "active");
  const counting = act && (act.id === "warmup" || act.id === "stream") && p.steps;
  const pct = p.steps ? Math.min(100, Math.round((p.written / p.steps) * 100)) : 0;
  $("#bar").className = "bar" + (running && !counting ? " indet" : "");
  $("#bar-fill").style.width = `${j.state === "done" ? 100 : pct}%`;
  $("#bar-text").textContent = running ? (counting ? `${pct}% · frame ${p.written} / ${p.steps}` : act ? act.label + "…" : "starting…")
    : j.state === "done" ? "done" : words[j.state] || "";
  // speed since the first frame step (warm-up start) and time left at that speed
  const t0 = (ph.find((x) => x.id === "warmup" && x.t0) || ph.find((x) => x.id === "stream" && x.t0) || {}).t0;
  const t1 = (ph.find((x) => x.id === "stream") || {}).t1 || (running ? now : j.ended);
  const rate = t0 && p.written ? p.written / Math.max(0.1, t1 - t0) : null;
  const left = rate && p.steps && running && counting ? (p.steps - p.written) / rate : null;
  const bits = [`elapsed ${mmss(j.elapsed_s)}`];
  if (p.steps) bits.unshift(`${p.written} / ${p.steps} frame files`);
  if (rate) bits.push(`${fmt(rate)} frames/s${meta.fps ? ` (${fmt(rate / meta.fps, 2)}× real time)` : ""}`);
  if (left != null) bits.push(`about ${mmss(left)} left`);
  if (state.unity && state.unity.mode === "broadcast" && running) bits.push(`Unity ${fmt(state.unity.behind_s)} s behind the cameras`);
  $("#job-stats").textContent = bits.join(" · ");
  const folder = meta.frames_dir || (j.run_dir ? `${j.run_dir}/unity_live_frames` : "");
  $("#job-folder").textContent = folder ? `frames: ${folder}` : "";
  $("#open-folder").hidden = !meta.run; $("#open-folder").dataset.run = meta.run || "";
  $("#job-err").textContent = j.error || "";
  $("#job-log").textContent = (j.log || []).join("\n");
  if (state.lastState === "running" && j.state !== "running") { loadRuns(); if (meta.run) loadTiming(meta.run); }
  if (!state.lastState && !running && meta.run && j.state === "done") loadTiming(meta.run);
  state.lastState = j.state;
}

$("#open-folder").addEventListener("click", async (ev) => {
  try { const d = await api("/api/open_folder", { run: ev.target.dataset.run }); toast(d.opened ? `Opened ${d.path}` : `Open it yourself: ${d.path}`, 6000); }
  catch (e) { toast(e.message, 6000); }
});

// ------------------------------------------------------------------ time report
async function loadTiming(name) {
  try { showTiming(await api(`/api/runs/${encodeURIComponent(name)}/timing`)); }
  catch (e) { toast("time report: " + e.message); }
}

function showTiming(t) {
  $("#timebox").hidden = false;
  $("#time-name").textContent = t.name;
  if (!t.has_report) {
    $("#time-body").innerHTML = `<p class="muted">No report yet${t.job_s ? ` (job ran ${mmss(t.job_s)})` : ""}: the numbers appear when the stream has finished.</p>`;
    return;
  }
  const ms = (v) => (v == null ? "–" : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${v.toFixed(v < 10 ? 1 : 0)} ms`);
  const e = t.e2e || {};
  const rows = [
    ["Whole job", `${mmss(t.job_s)} (setup + streaming + report)`],
    ["Setup", `${fmt(t.setup_s)} s: ${t.backend === "live" ? "2D models + opening the videos" : "loading the stored 2D"}${t.source_ready_s >= 0.05 ? ` (videos open after ${fmt(t.source_ready_s)} s)` : ""} — not counted as delay`],
    ["Streaming", `${t.written} / ${t.steps} frames in ${fmt(t.wall_s)} s for ${fmt(t.section_s)} s of video`],
    ["Speed", `${fmt(t.achieved_fps)} frames/s of ${fmt(t.target_fps, 0)} → ${fmt(t.realtime_factor, 2)}× real time`],
    ["Delay (capture → file)", `median ${ms(e.p50)}, p90 ${ms(e.p90)}, worst ${ms(e.max)}`],
    ["First frame file", `${fmt(t.first_frame_s, 2)} s after the clock started${t.first_2d_call_ms != null ? ` (first 2D call ${ms(t.first_2d_call_ms)})` : ""}`],
    ["Late fits after warm-up", t.late_after_warmup == null ? "–" : `${Math.round(t.late_after_warmup * 100)} %`],
    ["Dropped 2D frames", t.dropped_2d_frames ?? "–"],
    ["Report", t.report_s == null ? "–" : `${t.report_s} s`],
  ];
  let h = `<div class="tgrid">${rows.map(([k, v]) => `<div class="k">${k}</div><div class="v">${v}</div>`).join("")}</div>`;
  if (t.stages && t.stages.length) {
    h += `<table class="qt"><thead><tr><th>per frame</th><th>median</th><th>p90</th><th>worst</th></tr></thead><tbody>` +
      t.stages.map((s) => `<tr${s.key === "e2e" ? ' class="tot"' : ""}><td>${s.label}</td><td>${ms(s.p50)}</td><td>${ms(s.p90)}</td><td>${ms(s.max)}</td></tr>`).join("") +
      `</tbody></table>`;
  }
  $("#time-body").innerHTML = h;
}

function showUnity(u) {
  state.unity = u;
  const pill = $("#unity");
  if (u && u.in_play) {
    pill.className = "pill ok";
    pill.textContent = u.mode === "broadcast" ? "Unity: broadcasting live" : "Unity: in Play mode ✓";
    const what = { broadcast: `showing the live stream · ${u.received} frames · ${fmt(u.behind_s)} s behind the cameras · ${fmt(u.fps_in)} frames/s in`,
      replay: `showing a finished stream (${u.received} frames)`, clip: "showing a finished build", idle: "waiting" }[u.mode] || u.mode;
    $("#unity-detail").textContent = `In Play mode — ${what}${u.title ? " · " + u.title : ""}`;
  } else {
    pill.className = "pill bad";
    pill.textContent = "Unity: not in Play mode";
    $("#unity-detail").textContent = u && u.seen ? `Not in Play mode (last heartbeat ${fmt(u.age_s, 0)} s ago). Press Play in Unity.` : "No heartbeat yet: open the project in Unity and press Play.";
  }
}

async function pollJob() {
  try {
    const [d, u] = await Promise.all([api("/api/job"), api("/api/unity")]);
    showJob(d.current); showUnity(u);
    const running = d.current && (d.current.state === "running" || d.current.state === "starting");
    clearTimeout(state.polling);
    state.polling = setTimeout(pollJob, running ? 1000 : 2000);
  } catch (e) { clearTimeout(state.polling); state.polling = setTimeout(pollJob, 5000); }
}

// ------------------------------------------------------------------ past streams
function cls(v, good, ok) { return v === null || v === undefined ? "" : v <= good ? "good" : v <= ok ? "warnc" : "badc"; }

async function loadRuns() {
  try {
    const runs = await api("/api/runs");
    const tb = $("#runs tbody"); tb.innerHTML = "";
    if (!runs.length) tb.innerHTML = `<tr><td colspan="12" class="muted">No streams yet.</td></tr>`;
    runs.forEach((r) => {
      const s = r.summary || {};
      const tr = document.createElement("tr");
      tr.innerHTML = `<td><input type="checkbox" class="cmp" value="${r.name}"></td>
        <td class="name">${r.name}${r.layout === "live" ? ' <span class="muted">(old layout)</span>' : ""}</td>
        <td>${r.frames}${r.finished ? "" : " …"}</td><td>${fmt(s.achieved_fps)}</td><td>${fmt(s.realtime_factor, 2)}</td>
        <td>${s.late_after_warmup == null ? "–" : Math.round(s.late_after_warmup * 100) + "%"}</td>
        <td class="${cls(s.mpjpe_cm, 3.2, 3.5)}">${fmt(s.mpjpe_cm)}</td>
        <td class="${cls(s.knee_mae_deg, 6.5, 7.3)}">${fmt(s.knee_mae_deg)}</td>
        <td class="${s.swaps ? "badc" : s.swaps === 0 ? "good" : ""}">${s.swaps ?? "–"}</td>
        <td>${fmt(s.setup_s)}</td><td>${s.delay_p50_ms == null ? "–" : fmt(s.delay_p50_ms / 1000, 2)}</td>
        <td><div class="acts"><button class="small" data-play="${r.name}" title="replay this stream in Unity">Replay</button>
          ${r.has_report ? `<button class="small ghost" data-time="${r.name}" title="time report of this stream">Time</button>` : ""}
          ${r.has_report ? `<button class="small ghost" data-report="${r.name}">Report</button>` : ""}</div></td>`;
      tb.appendChild(tr);
    });
    $("#compare-btn").disabled = true;
  } catch (e) { toast("streams: " + e.message); }
}
$("#refresh-runs").addEventListener("click", loadRuns);
$("#past").addEventListener("toggle", () => { if ($("#past").open) loadRuns(); });
document.addEventListener("change", (ev) => {
  if (ev.target.classList && ev.target.classList.contains("cmp"))
    $("#compare-btn").disabled = document.querySelectorAll(".cmp:checked").length < 2;
});
$("#compare-close").addEventListener("click", () => ($("#compare-box").hidden = true));
$("#report-close").addEventListener("click", () => ($("#report-box").hidden = true));
$("#compare-btn").addEventListener("click", async () => {
  const names = [...document.querySelectorAll(".cmp:checked")].map((x) => x.value);
  try {
    const d = await api(`/api/compare?runs=${names.map(encodeURIComponent).join(",")}`);
    const rows = d.rows, base = rows[0].summary || {};
    const metric = [["Output fps", "achieved_fps", 1, null], ["× real time", "realtime_factor", 2, null],
      ["Setup s", "setup_s", 1, null], ["Delay median ms", "delay_p50_ms", 0, null],
      ["Late after warm-up", "late_after_warmup", 0, null], ["MPJPE cm", "mpjpe_cm", 2, 0.5], ["PA-MPJPE cm", "pa_mpjpe_cm", 2, 0.5],
      ["Knee error °", "knee_mae_deg", 1, 1.0], ["Swaps", "swaps", 0, 0]];
    let h = `<thead><tr><th></th>${rows.map((r, i) => `<th>${i ? "" : "baseline<br>"}${r.name}</th>`).join("")}</tr></thead><tbody>`;
    metric.forEach(([lab, k, dp, budget]) => {
      h += `<tr><td class="metric">${lab}</td>`;
      rows.forEach((r, i) => {
        const v = (r.summary || {})[k];
        let txt = k === "late_after_warmup" ? (v == null ? "–" : Math.round(v * 100) + "%") : fmt(v, dp);
        let c = "";
        if (i && budget !== null && v != null && base[k] != null) {
          const dl = v - base[k];
          c = k === "swaps" ? (v > 0 ? "badc" : "good") : dl <= budget ? "good" : "badc";
          if (k !== "swaps") txt += ` <span class="delta ${c}">(${dl >= 0 ? "+" : ""}${dl.toFixed(dp)})</span>`;
        } else if (k === "swaps" && v != null) c = v > 0 ? "badc" : "good";
        h += `<td class="${c}">${txt}</td>`;
      });
      h += "</tr>";
    });
    if (d.differs.length) {
      h += `<tr><td class="metric" colspan="${rows.length + 1}">Settings that differ</td></tr>`;
      d.differs.forEach((f) => { h += `<tr><td class="metric">${f.label}</td>${rows.map((r) => `<td>${JSON.stringify(r.settings[f.key])}</td>`).join("")}</tr>`; });
    } else h += `<tr><td class="metric" colspan="${rows.length + 1}">Same settings (repeat runs: the difference is noise)</td></tr>`;
    $("#compare").innerHTML = h + "</tbody>"; $("#compare-box").hidden = false;
  } catch (e) { toast(e.message, 6000); }
});
document.addEventListener("click", async (ev) => {
  const t = ev.target;
  try {
    if (t.dataset && t.dataset.play) { const d = await api("/api/play", { kind: "run", name: t.dataset.play }); toast(`Unity replays ${d.title}`); }
    if (t.dataset && t.dataset.time) { await loadTiming(t.dataset.time); $("#timebox").scrollIntoView({ behavior: "smooth", block: "nearest" }); }
    if (t.dataset && t.dataset.report) {
      $("#report").textContent = await (await fetch(`/api/runs/${encodeURIComponent(t.dataset.report)}/report`)).text();
      $("#report-title").textContent = t.dataset.report; $("#report-box").hidden = false;
    }
  } catch (e) { toast(e.message, 6000); }
});

// ------------------------------------------------------------------ start
(async function init() {
  try {
    [state.info, state.schema, state.clips] = await Promise.all([api("/api/info"), api("/api/schema"), api("/api/clips")]);
    $("#machine").textContent = state.info.machine === "laptop-cached" ? "laptop: stored 2D" : `GPU: ${state.info.machine}`;
    $("#machine").title = `machine preset ${state.info.machine}\nUnity project ${state.info.unity_project}\nruns ${state.info.runs_dir}`;
    const s = $("#clip"); s.innerHTML = "";
    state.clips.forEach((c) => { const o = document.createElement("option"); o.value = c.name; o.textContent = c.name; s.appendChild(o); });
    await loadPresets();
    pollJob();
    api("/api/runs").then((runs) => { const r = runs.find((x) => x.has_report); if (r && $("#timebox").hidden) loadTiming(r.name); }).catch(() => {});
  } catch (e) { toast("could not reach the server: " + e.message, 8000); }
})();
