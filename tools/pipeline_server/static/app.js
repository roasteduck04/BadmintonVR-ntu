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
    [a, b].forEach((x) => { x.type = "number"; x.step = "any"; x.style.width = "90px"; });
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
  return wrap;
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
  state.values[key] = v;
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

function showJob(j) {
  const box = $("#job");
  box.className = "job " + (j ? j.state : "idle");
  const running = !!(j && (j.state === "running" || j.state === "starting"));
  $("#start").disabled = running; $("#stop").disabled = !running;
  $("#start").textContent = running ? "Streaming…" : "Start streaming to Unity";
  if (!j) { $("#job-title").textContent = "Idle"; $("#job-stats").textContent = ""; $("#job-folder").textContent = ""; return; }
  const words = { starting: "Starting", running: "Streaming", done: "Stream finished", failed: "Failed", stopped: "Stopped" };
  $("#job-title").textContent = `${words[j.state] || j.state}: ${j.title}`;
  const p = j.progress || {};
  $("#bar-fill").style.width = `${p.steps ? Math.round((p.written / p.steps) * 100) : 0}%`;
  const fps = j.elapsed_s > 0 && p.written ? p.written / j.elapsed_s : null;
  $("#job-stats").textContent = p.steps ? `${p.written} / ${p.steps} frame files written · ${fmt(j.elapsed_s, 0)} s · ${fmt(fps)} frames/s`
    : `${fmt(j.elapsed_s, 0)} s · loading the models and opening the videos…`;
  $("#job-folder").textContent = j.run_dir ? `folder: ${j.run_dir.split(/[\\/]/).slice(-2).join("/")}/unity_live_frames/` : "";
  $("#job-err").textContent = j.error || "";
  $("#job-log").textContent = (j.log || []).join("\n");
  if (state.lastState === "running" && j.state !== "running") loadRuns();
  state.lastState = j.state;
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
    if (!runs.length) tb.innerHTML = `<tr><td colspan="10" class="muted">No streams yet.</td></tr>`;
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
        <td><div class="acts"><button class="small" data-play="${r.name}" title="replay this stream in Unity">Replay</button>
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
  } catch (e) { toast("could not reach the server: " + e.message, 8000); }
})();
