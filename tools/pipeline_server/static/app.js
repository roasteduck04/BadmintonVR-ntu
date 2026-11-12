// BadmintonVR live page (v0.8.6): sidebar steps 1 Source, 2 Calibrate (calib.js), 3 Stream. Plain JS, no build step.
"use strict";

const $ = (s) => document.querySelector(s);
const state = { schema: null, base: {}, values: {}, clips: [], presets: [], info: null, polling: null, unity: null,
  input: null, camCalib: {}, step: "source" };

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
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
// back_051 -> "Back", backleft_11 -> "Back left"; live camera names stay as they are
function pretty(v) {
  const m = /^(back|front)(left|right)?_\d+$/.exec(v);
  return m ? (m[1][0].toUpperCase() + m[1].slice(1)) + (m[2] ? " " + m[2] : "") : v;
}
const currentClip = () => (state.input && state.input.clip) || (state.clips[0] && state.clips[0].name);
const clipOf = (name) => state.clips.find((c) => c.name === name);
const usedCams = () => ((state.input && state.input.cameras) || []).filter((c) => c.use && c.url);
function chosenViews(c) {
  const inp = state.input || {};
  return inp.clip === c.name && inp.views && inp.views.length ? inp.views : c.live_views;
}
function streamBlocked() { const c = clipOf(currentClip()); return !!(c && !c.streamable); }
function calChip(rms, calibrated) {
  if (!calibrated) return `<span class="chip">not calibrated</span>`;
  if (rms == null) return `<span class="chip ok">calibrated</span>`;
  return `<span class="chip ${rms > 15 ? "warn" : "ok"}">${fmt(rms)} px</span>`;
}

// ------------------------------------------------------------------ steps
function showStep(step) {
  state.step = step;
  document.querySelectorAll("#steps .step").forEach((x) => x.classList.toggle("on", x.dataset.step === step));
  document.querySelectorAll(".panel").forEach((p) => (p.hidden = p.id !== "step-" + step));
  if (step !== "source") stopLive();
  if (step === "source") loadSource();
  if (step === "calibrate") Calib.open();
  if (step === "stream") loadSettings();          // a new calibration / sync file shows in the clip line
}
document.querySelectorAll("#steps .step").forEach((b) => b.addEventListener("click", () => showStep(b.dataset.step)));

function sidebar() {
  const inp = state.input || {}, c = clipOf(currentClip());
  if (!c) return;
  const cam = inp.kind === "camera";
  const views = cam ? usedCams().map((x) => x.name) : chosenViews(c);
  $("#st-source").textContent = `${c.label.replace(/ \(.*\)$/, "")} · ${views.length} ${cam ? "cam" : "angle"}${views.length === 1 ? "" : "s"}`;
  const ncal = views.filter((v) => (cam ? state.camCalib[v] : c.calibrated[v])).length;
  $("#st-calib").textContent = `${ncal} / ${views.length}`;
}

// ------------------------------------------------------------------ 1 Source
const src = { kind: "video", clip: null, views: {}, cams: [], calib: {} };   // the form while editing

async function loadSource() {
  try {
    [state.input, state.clips] = await Promise.all([api("/api/input"), api("/api/clips")]);
  } catch (e) { toast("source: " + e.message); return; }
  const inp = state.input;
  src.kind = inp.kind; src.clip = inp.clip;
  src.views = { [inp.clip]: inp.views && inp.views.length ? [...inp.views] : null };
  src.cams = (inp.cameras && inp.cameras.length ? inp.cameras : [1, 2, 3].map((i) => ({ name: `cam${i}`, url: "", use: true })))
    .map((c) => ({ ...c }));
  $("#src-clip").innerHTML = state.clips.map((c) => `<option value="${c.name}">${esc(c.label)}${c.streamable ? "" : " — calibrate only"}</option>`).join("");
  $("#src-clip").value = src.clip;
  $("#src-fps").value = inp.fps; $("#src-mins").value = inp.max_minutes;
  $("#src-msg").textContent = "";
  await camCalibStatus();
  renderSourceKind(); await renderClip();
  sidebar(); testcamsStatus(); usbStatus();
}

async function camCalibStatus() {
  // calibration of the live cameras in use (by name), for the table and the sidebar
  if ((state.input || {}).kind !== "camera") { state.camCalib = {}; return; }
  try {
    const cams = await api("/calib/api/source-cameras");
    state.camCalib = Object.fromEntries(cams.filter((c) => c.calibrated).map((c) => [c.view, c.rms ?? 0]));
  } catch (e) { state.camCalib = {}; }
}

function renderSourceKind() {
  document.querySelectorAll("#src-kind .segb").forEach((b) => b.classList.toggle("on", b.dataset.kind === src.kind));
  $("#src-video").hidden = src.kind !== "video"; $("#src-camera").hidden = src.kind !== "camera";
  renderCams();
}
document.querySelectorAll("#src-kind .segb").forEach((b) => b.addEventListener("click", () => { src.kind = b.dataset.kind; renderSourceKind(); }));

async function renderClip() {
  const c = clipOf(src.clip);
  if (!c) return;
  $("#src-clip-note").textContent = `${c.mode === "doubles" ? "Doubles" : "Singles"}, ${c.views.length} cameras recorded.` +
    (c.streamable ? "" : " Calibrate only for now: the live tracker handles singles (doubles is a separate piece of work).");
  const want = src.views[c.name] || c.live_views;
  let calib = {};
  try { (await api(`/calib/api/cameras?clip=${encodeURIComponent(c.name)}`)).forEach((x) => (calib[x.view] = x)); } catch (e) { /* statuses only */ }
  $("#src-angles").innerHTML = c.views.map((v) => {
    const on = want.includes(v), k = calib[v] || {};
    return `<label class="angle${on ? " on" : ""}"><input type="checkbox" data-view="${v}"${on ? " checked" : ""}>
      <img class="thumb" loading="lazy" alt="" src="/calib/thumbs/${encodeURIComponent(v)}.jpg">
      <div class="arow"><span class="aname">${pretty(v)}</span>${calChip(k.rms, k.calibrated)}</div></label>`;
  }).join("");
  $("#src-angles").querySelectorAll("input").forEach((i) => i.addEventListener("change", () => {
    i.closest(".angle").classList.toggle("on", i.checked);
    src.views[c.name] = [...$("#src-angles").querySelectorAll("input:checked")].map((x) => x.dataset.view);
  }));
  $("#src-angles").querySelectorAll("img").forEach((im) => im.addEventListener("error", () => (im.style.visibility = "hidden"), { once: true }));
  try {
    const vars = await api(`/api/input/variants?clip=${encodeURIComponent(c.name)}`);
    $("#src-variant").innerHTML = [`<option value="">as recorded</option>`]
      .concat((vars.variants || []).map((t) => `<option value="${t}">${t} copies</option>`)).join("");
  } catch (e) { $("#src-variant").innerHTML = `<option value="">as recorded</option>`; }
  $("#src-variant").value = (state.input && state.input.video_variant) || "";
}
$("#src-clip").addEventListener("change", () => { src.clip = $("#src-clip").value; renderClip(); });

// v0.8.7 USB cameras on this PC: what "Find USB cameras" saw ({index: {name, size, fps, thumb, in_use}})
const usb = { found: {}, where: null };
const usbLabel = (i) => {
  const f = usb.found[i];
  return `USB camera ${i}` + (f && f.name ? ` · ${f.name}` : "") + (f && f.size ? ` (${f.size[0]}×${f.size[1]})` : "");
};
function addrCell(c, i) {
  if (c.type !== "usb") {
    return `<input class="addr" data-i="${i}" data-f="url" value="${esc(c.url)}" placeholder="rtsp://… or http://…/stream.mjpg">`;
  }
  const opts = [...Array(10).keys()].map((d) => `<option value="${d}"${Number(c.device) === d ? " selected" : ""}>${esc(usbLabel(d))}</option>`).join("");
  return `<select class="dev" data-i="${i}" data-f="device">${opts}</select>`
    + (c.url ? `<div class="tiny muted" title="the server serves this USB camera as a stream here">${esc(c.url)}</div>` : "");
}
function renderCams() {
  $("#src-cams tbody").innerHTML = src.cams.map((c, i) => `<tr>
    <td><input type="checkbox" data-i="${i}" data-f="use"${c.use ? " checked" : ""}></td>
    <td><input class="nm" data-i="${i}" data-f="name" value="${esc(c.name)}" maxlength="40"></td>
    <td><select class="typ" data-i="${i}" data-f="type"><option value="network">Network</option>
      <option value="usb"${c.type === "usb" ? " selected" : ""}>USB on this PC</option></select></td>
    <td>${addrCell(c, i)}</td>
    <td><input type="checkbox" data-i="${i}" data-f="rotate"${c.rotate ? " checked" : ""} title="this camera hangs upside down: turn its picture 180°"></td>
    <td class="test" data-test="${i}">–</td>
    <td>${c.name in state.camCalib ? calChip(state.camCalib[c.name], true) : calChip(null, false)}</td>
    <td><button class="btn small" data-del="${i}" title="remove this camera">×</button></td></tr>`).join("");
}
$("#src-cams").addEventListener("input", (ev) => {
  const t = ev.target, i = Number(t.dataset.i), f = t.dataset.f;
  if (!f || Number.isNaN(i)) return;
  const c = src.cams[i];
  if (f === "type") {                               // Network <-> USB: the address cell changes
    c.type = t.value; c.url = "";
    if (c.type === "usb") c.device = c.device ?? firstFreeDevice(); else delete c.device;
    renderCams(); return;
  }
  if (f === "device") { c.device = Number(t.value); c.url = ""; renderCams(); return; }
  if (f === "rotate") { if (t.checked) c.rotate = 180; else delete c.rotate; return; }
  c[f] = t.type === "checkbox" ? t.checked : t.value.trim();
});
function firstFreeDevice() {
  const taken = new Set(src.cams.filter((c) => c.type === "usb").map((c) => Number(c.device)));
  const seen = Object.keys(usb.found).map(Number).sort((a, b) => a - b);      // found ones first
  return seen.find((d) => !taken.has(d)) ?? [...Array(10).keys()].find((d) => !taken.has(d)) ?? 0;
}
$("#src-cams").addEventListener("click", (ev) => {
  const d = ev.target.dataset && ev.target.dataset.del;
  if (d === undefined) return;
  src.cams.splice(Number(d), 1); renderCams();
});
$("#src-add").addEventListener("click", () => {
  let n = src.cams.length + 1;
  while (src.cams.some((c) => c.name === `cam${n}`)) n++;
  src.cams.push({ name: `cam${n}`, url: "", use: true }); renderCams();
});
$("#src-test").addEventListener("click", async () => {
  const rows = src.cams.filter((c) => c.url || (c.type === "usb" && c.use));
  const names = new Set(rows.map((c) => c.name));
  const anyUsb = rows.some((c) => c.type === "usb");
  document.querySelectorAll("#src-cams [data-test]").forEach((td) => {
    td.className = "test";
    td.textContent = names.has((src.cams[Number(td.dataset.test)] || {}).name) ? (anyUsb ? "starting the USB cameras / testing…" : "testing…") : "–";
  });
  try {
    const r = await api("/api/input/test", { rows, fps: Number($("#src-fps").value) });
    let moved = false;
    Object.entries(r).forEach(([v, x]) => {
      const c = src.cams.find((k) => k.name === v);
      if (c && x.url && c.url !== x.url) { c.url = x.url; moved = true; }
    });
    if (moved) renderCams();                        // shows the USB cameras' stream addresses
    Object.entries(r).forEach(([v, x]) => {
      const td = document.querySelector(`#src-cams [data-test="${src.cams.findIndex((k) => k.name === v)}"]`);
      if (!td) return;
      td.className = "test " + (x.ok ? "good" : "badc");
      td.textContent = x.ok ? `✓ ${x.size[0]}×${x.size[1]} (${x.ms} ms)` : `✗ ${x.error}`;
      if (x.thumb) { const img = document.createElement("img"); img.src = x.thumb; img.className = "cam-thumb"; td.append(img); }
    });
    if (anyUsb) usbStatus();
  } catch (e) {
    toast(e.message, 12000);
    document.querySelectorAll("#src-cams [data-test]").forEach((td) => (td.textContent = "–"));
  }
});

// v0.8.7 what the feed really does: Check the feed (15 s of reading, like the engine) and Measure delay (the page
// flashes, the server times when each camera shows it)
const feedRows = () => src.cams.filter((c) => c.url || c.type === "usb");
function showFeed(r, what) {
  const rows = Object.entries(r.cameras).map(([n, c]) => {
    const s = c.stats || {};
    if (!s.fps) return `<tr><td><b>${esc(n)}</b></td><td colspan="5" class="badc">${esc(c.error || "no frames arrived")}</td></tr>`;
    const low = s.fps < 0.9 * r.fps;
    const delay = c.delays_s ? (c.delay_s != null ? `<b>${c.delay_s} s</b> <span class="tiny muted">(${c.delay_min_s}–${c.delay_max_s}, ${c.flashes_seen} of ${c.delays_s.length} flashes)</span>`
      : '<span class="muted">screen not seen</span>') : "–";
    return `<tr><td><b>${esc(n)}</b><div class="tiny muted">${c.size ? c.size[0] + "×" + c.size[1] : ""} ${esc(c.codec || "")}</div></td>
      <td class="${low ? "badc" : "good"}">${s.fps} fps<div class="tiny muted">worst second ${s.fps_worst_second}</div></td>
      <td>${s.gap_ms_p50} ms<div class="tiny muted">1 in 20 above ${s.gap_ms_p95} · longest ${s.gap_ms_max}</div></td>
      <td>${delay}</td>
      <td>${c.verdict.map((v) => `<div class="${v === "steady" ? "good" : "warnc"}">${esc(v)}</div>`).join("")}</td></tr>`;
  }).join("");
  $("#src-feed").innerHTML = `<div class="small"><b>${esc(what)}</b> · ${r.fps} fps expected · even spacing = ${(1000 / r.fps).toFixed(1)} ms</div>
    <div class="table-wrap"><table class="tbl"><thead><tr><th>Camera</th><th>Arriving</th><th>Time between frames</th><th>Delay</th><th>Findings</th></tr></thead>
    <tbody>${rows}</tbody></table></div><div class="tiny muted">Report and one picture per camera: ${esc(r.folder)}</div>`;
}
$("#src-check").addEventListener("click", async () => {
  const rows = feedRows();
  if (!rows.length) { toast("Add a camera address first."); return; }
  const b = $("#src-check"); b.disabled = true; b.textContent = "Checking (15 s)…";
  try { showFeed(await api("/api/input/check", { rows, fps: Number($("#src-fps").value), seconds: 15 }), "Feed check, 15 s"); }
  catch (e) { toast(e.message, 12000); }
  finally { b.disabled = false; b.textContent = "Check the feed"; }
});
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
$("#src-delay").addEventListener("click", async () => {
  const rows = feedRows();
  if (!rows.length) { toast("Add a camera address first."); return; }
  if (!confirm("Aim one or more cameras at this screen (the closer it fills the picture, the better). The page will flash white and black for about 40 seconds.")) return;
  const b = $("#src-delay"), fl = $("#flash"); b.disabled = true;
  try {
    const s = await api("/api/input/delay/start", { rows, fps: Number($("#src-fps").value) });
    if (!s.cameras.length) throw new Error("no camera sends a picture: " + Object.entries(s.failed).map(([n, e]) => `${n} ${e}`).join(", "));
    fl.className = ""; fl.hidden = false;
    await pause(3000);
    let white = false;
    for (let i = 0; i < 10; i++) {
      white = !white;
      fl.className = white ? "white" : "";
      await new Promise((r) => requestAnimationFrame(r));
      await api("/api/input/delay/flash", { white });
      await pause(3000 + Math.random() * 1000);
    }
    fl.hidden = true;
    showFeed(await api("/api/input/delay/finish", {}), "Delay test, 10 flashes");
  } catch (e) { fl.hidden = true; toast(e.message, 12000); api("/api/input/delay/finish", {}).catch(() => {}); }
  finally { b.disabled = false; }
});

// v0.8.7 live preview: every camera row with an address plays in the page (small, a few pictures a second, each
// labelled with the frame rate that really arrives) while the cameras are set up and tested
let liveOn = false;
function stopLive() {
  if (!liveOn) return;
  liveOn = false;
  $("#src-live-grid").innerHTML = ""; $("#src-live").textContent = "Live preview";
  api("/api/input/preview/stop", {}).catch(() => {});
}
$("#src-live").addEventListener("click", async () => {
  if (liveOn) { stopLive(); return; }
  const rows = src.cams.filter((c) => c.url || c.type === "usb");
  if (!rows.length) { toast("Add a camera address first."); return; }
  const b = $("#src-live"); b.disabled = true;
  try {
    const r = await api("/api/input/preview", { rows, fps: Number($("#src-fps").value) });
    liveOn = true; b.textContent = "Stop preview";
    $("#src-live-grid").innerHTML = r.cameras.map((n) =>
      `<figure><img src="/api/input/preview/${encodeURIComponent(n)}.mjpg?e=${r.epoch}" alt="${esc(n)}"><figcaption class="tiny muted">${esc(n)}</figcaption></figure>`).join("");
  } catch (e) { toast(e.message, 12000); }
  finally { b.disabled = false; }
});

// v0.8.7 USB cameras on this PC: Find lists them (preview each), "+ Use" adds a row; the relay status line
function renderUsbFound() {
  const list = $("#src-usb-list"), cams = Object.values(usb.found).sort((a, b) => a.index - b.index);
  if (!cams.length) { list.innerHTML = ""; return; }
  const inTable = (d) => src.cams.find((c) => c.type === "usb" && Number(c.device) === d);
  list.innerHTML = cams.map((f) => {
    const row = inTable(f.index);
    const what = f.in_use ? `in use: ${esc(f.in_use)}` : `${f.size ? f.size[0] + "×" + f.size[1] : "?"}${f.fps ? " · " + f.fps + " fps" : ""}`;
    return `<div class="usb-card">${f.thumb ? `<img src="${f.thumb}" alt="">` : `<div class="usb-nothumb">${f.in_use ? "streaming" : "no preview"}</div>`}
      <div class="small"><b>${f.index}</b>${f.name ? " · " + esc(f.name) : ""}</div><div class="tiny muted">${what}</div>
      ${row ? `<div class="tiny">= ${esc(row.name)}</div>` : `<button class="btn small" data-usb-add="${f.index}">+ Use</button>`}</div>`;
  }).join("");
}
$("#src-usb-list").addEventListener("click", (ev) => {
  const d = ev.target.dataset && ev.target.dataset.usbAdd;
  if (d === undefined) return;
  let n = src.cams.length + 1;
  while (src.cams.some((c) => c.name === `cam${n}`)) n++;
  const blank = src.cams.findIndex((c) => c.type !== "usb" && !c.url);       // fill an empty row first
  const row = { name: blank >= 0 ? src.cams[blank].name : `cam${n}`, url: "", use: true, type: "usb", device: Number(d) };
  if (blank >= 0) src.cams[blank] = row; else src.cams.push(row);
  renderCams(); renderUsbFound();
  $("#src-msg").textContent = `${row.name} = USB camera ${d}. Test cameras, then Save source.`;
});
$("#src-usb-find").addEventListener("click", async () => {
  const b = $("#src-usb-find");
  b.disabled = true;
  $("#src-usb-list").innerHTML = "";
  $("#src-usb-note").textContent = "Looking for USB cameras (each one is opened for a frame; a few seconds)…";
  try {
    const r = await api("/api/usb/find", { fps: Number($("#src-fps").value) });
    usb.where = r.where;
    usb.found = Object.fromEntries(r.cameras.map((f) => [f.index, f]));
    renderCams(); renderUsbFound();
    await usbStatus(r.cameras.length ? null
      : `No USB camera opens${r.where === "Windows" ? " on Windows" : ""} (plugged in? another app using it?).`);
  } catch (e) { toast(e.message, 15000); $("#src-usb-note").textContent = ""; }
  finally { b.disabled = false; }
});
async function usbStatus(extra) {
  let s;
  try { s = await api("/api/usb"); } catch (e) { return; }      // older server
  const note = $("#src-usb-note");
  const parts = [];
  if (extra) parts.push(esc(extra));
  if (s.running) {
    const devs = Object.entries(s.devices).map(([nm, d]) => `${esc(nm)} = camera ${d}`).join(", ");
    parts.push(`<b>USB cameras streaming</b> (${esc(s.where)}, ${s.fps} fps asked): ${devs}.
      <button class="btn small" id="src-usb-stop" title="frees the USB cameras for other apps; Save, Test or Warm up starts them again">Stop USB cameras</button>`);
    if (s.error) parts.push(`<b class="badc">${esc(s.error)}</b>`);
  } else if (s.error) {
    parts.push(`USB cameras stopped: ${esc(s.error)}`);
  } else if (Object.keys(usb.found).length) {
    parts.push(`USB cameras found${usb.where === "Windows" ? " on Windows (this server runs in WSL)" : ""}: pick one with <b>+ Use</b> or the Type column.`);
  }
  note.innerHTML = parts.join(" ");
  const stop = $("#src-usb-stop");
  if (stop) stop.addEventListener("click", async () => {
    try { await api("/api/usb/stop", {}); } catch (e) { toast(e.message, 7000); }
    usbStatus();
  });
}
// v0.8.7 one-click test cameras: the server plays the clip's ticked angles as looping cameras and saves them as the source
function renderTestcams(s) {
  const note = $("#src-testcams-note"), b = $("#src-testcams");
  if (s && s.running) {
    const c = clipOf(s.clip);
    note.innerHTML = `<b>Test cameras on</b>: ${s.views.map(pretty).join(", ")} - ${esc(c ? c.label : s.clip)}${s.variant ? ` (${esc(s.variant)} copies)` : ""} looping from ${s.start_s} s.
      <button class="btn small" id="src-testcams-stop">Stop test cameras</button>`;
    $("#src-testcams-stop").addEventListener("click", async () => {
      try { renderTestcams(await api("/api/testcams/stop", {})); } catch (e) { toast(e.message, 7000); }
    });
    b.textContent = "▶ Restart test cameras";
  } else {
    note.textContent = s && s.error ? `Test cameras stopped: ${s.error}` : "";
    b.textContent = "▶ Test cameras from the clip";
  }
}
async function testcamsStatus() {
  try { renderTestcams(await api("/api/testcams")); } catch (e) { /* older server */ }
}
$("#src-testcams").addEventListener("click", async () => {
  const c = clipOf(src.clip), b = $("#src-testcams");
  if (!c) return;
  b.disabled = true;
  $("#src-testcams-note").textContent = "Starting the test cameras (opening the videos)…";
  try {
    const r = await api("/api/testcams/start", { clip: src.clip, views: src.views[src.clip] || c.live_views,
      variant: $("#src-variant").value || null });                // Video size: smaller copies decode faster
    state.input = r.input;
    src.kind = "camera"; src.cams = r.input.cameras.map((x) => ({ ...x }));
    $("#src-fps").value = r.input.fps;
    await camCalibStatus(); renderSourceKind(); sidebar(); renderTestcams(r.status);
    $("#src-msg").textContent = `Saved: ${r.status.views.length} test camera${r.status.views.length === 1 ? "" : "s"} `
      + "(the clip's calibrations apply). Warm up again in step 3.";
    $("#src-test").click();                         // a frame from each camera
  } catch (e) { toast(e.message, 8000); $("#src-testcams-note").textContent = ""; }
  finally { b.disabled = false; }
});

$("#src-save").addEventListener("click", async () => {
  const c = clipOf(src.clip);
  const body = { kind: src.kind, clip: src.clip, views: src.views[src.clip] || c.live_views,
    video_variant: $("#src-variant").value || null, cameras: src.cams, fps: Number($("#src-fps").value), max_minutes: Number($("#src-mins").value) };
  const usbRows = src.kind === "camera" && src.cams.some((c) => c.type === "usb" && c.use);
  $("#src-msg").textContent = usbRows ? "Saving (starting the USB cameras)…" : "";
  try {
    state.input = await api("/api/input", body);
    src.cams = (state.input.cameras || []).map((x) => ({ ...x }));              // USB rows now carry their address
    await camCalibStatus(); renderCams(); sidebar(); usbStatus();
    const what = state.input.kind === "camera" ? `${usedCams().length} live camera${usedCams().length === 1 ? "" : "s"}`
      : `${c.label}, ${state.input.views.length} angle${state.input.views.length === 1 ? "" : "s"}`;
    $("#src-msg").textContent = `Saved: ${what}. Calibrate them (step 2), then warm up again (step 3).`;
  } catch (e) { toast(e.message, usbRows ? 15000 : 7000); $("#src-msg").textContent = ""; }
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
  const cam = state.input && state.input.kind === "camera";
  const src = cam ? `live camera streams (${usedCams().length} cameras, step 1)`
    : v.backend === "live" ? "straight from the videos (2D pose on the GPU, frame by frame)" : "the stored 2D files (this machine has no NVIDIA GPU)";
  $("#source-line").innerHTML = v.output_fps ? `<b>Source:</b> ${src} · ${v.output_fps} fps · ${v.pace === "realtime" ? "real time (camera clock)" : "as fast as possible"}`
    : `<b>Source:</b> ${src}`;
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

async function loadSettings(fresh) {       // fresh: drop the values changed on the page (a preset was picked)
  try { state.input = await api("/api/input"); } catch (e) { /* keep the last one */ }
  const clip = currentClip();
  if (!clip) return;
  const c = state.clips.find((x) => x.name === clip);
  if (!state.presetPicked) {       // doubles: 30 fps unless a preset was picked (60 fps doubles is 2D-bound, round 7)
    const want = c && c.mode === "doubles" && state.presets.some((p) => p.name === "live_30") ? "live_30"
      : state.info.default_settings;
    if (state.presets.some((p) => p.name === want)) $("#preset").value = want;
  }
  const preset = $("#preset").value;
  const blocked = c && !c.streamable;
  $("#stream-blocked").hidden = !blocked;
  if (blocked) $("#stream-blocked").innerHTML = `<b>${c.label}</b> can be calibrated but not streamed yet: ${c.mode === "doubles"
    ? "the live player tracker handles singles only (the doubles tracker is a separate piece of work)" : "it has no live config"}.
    Pick a singles clip in step 1 to stream.`;
  $("#warm").disabled = !!blocked;
  if (blocked) { $("#groups").innerHTML = ""; $("#clip-info").textContent = ""; sourceLine(); return; }
  try {
    const d = await api(`/api/settings?clip=${encodeURIComponent(clip)}&preset=${encodeURIComponent(preset)}`);
    const key = clip + "|" + preset, keep = {};      // same clip + preset as shown: the values changed here stay
    if (!fresh && state.settingsKey === key && state.values && state.base)
      Object.keys(state.values).forEach((k) => {
        if (JSON.stringify(state.values[k]) !== JSON.stringify(state.base[k])) keep[k] = state.values[k];
      });
    state.base = d.values; state.values = Object.assign(JSON.parse(JSON.stringify(d.values)), keep);
    state.settingsKey = key;
    renderForm();
  } catch (e) { toast("settings: " + e.message); }
  const p = state.presets.find((x) => x.name === preset);
  $("#preset-about").textContent = p ? p.about : "";
  if (c) {
    const cam = state.input && state.input.kind === "camera";
    const views = cam ? usedCams().map((x) => x.name) : chosenViews(c);
    const ncal = views.filter((v) => c.calibrated[v] || (cam && state.camCalib[v])).length;
    const bits = [`${c.label}: ${views.length} ${cam ? "camera" : "angle"}${views.length === 1 ? "" : "s"} (${views.map(pretty).join(", ")})`,
      !cam && c.section_s ? `section ${c.section_s[0]}–${c.section_s[1] ?? "end"} s` : "",
      ncal === views.length ? "all calibrated" : `${views.length - ncal} not calibrated (step 2)`,
      views.length === 1 ? "1-camera mode: court ~10–15 cm, rough pose, no racket / shuttle 3D (heights: One camera)"
        : cam ? "offsets measured in the warm-up" : c.sync_file ? "sync file present" : "no sync file"];
    $("#clip-info").textContent = bits.filter(Boolean).join(" · ");
  }
}

async function loadPresets(select) {
  state.presets = await api("/api/presets");
  const s = $("#preset"); s.innerHTML = "";
  state.presets.forEach((p) => { const o = document.createElement("option"); o.value = p.name; o.textContent = p.name + (p.builtin ? " (built in)" : ""); s.appendChild(o); });
  s.value = select || state.info.default_settings;
  if (select) state.presetPicked = true;
  await loadSettings();
}

$("#preset").addEventListener("change", () => { state.presetPicked = true; loadSettings(true); });
$("#reset").addEventListener("click", () => { state.values = JSON.parse(JSON.stringify(state.base)); renderForm(); });
$("#save").addEventListener("click", async () => {
  const name = $("#save-name").value.trim().toLowerCase();
  try {
    await api("/api/presets", { name, values: state.values, about: `saved from the page for ${currentClip()}` });
    await loadPresets(name); toast(`Preset ${name} saved (configs/live/settings/${name}.yaml).`);
  } catch (e) { toast(e.message, 6000); }
});

// ------------------------------------------------------------------ start / stop
async function start(force) {
  $("#confirm").hidden = true;
  try {
    $("#start").disabled = true;
    const j = await api("/api/stream/start", { clip: currentClip(), preset: $("#preset").value, values: state.values, force: !!force });
    showJob(j);
    toast(state.unity && state.unity.in_play ? "Streaming — Unity shows each pose as its file lands." : "Streaming — Unity can join any time (press Play).", 5000);
    pollJob();
  } catch (e) {
    $("#start").disabled = false;
    if (e.detail && e.detail.code === "unity_not_playing") { $("#confirm-msg").textContent = e.detail.message; $("#confirm").hidden = false; }
    else toast(e.message, 9000);
  }
}

async function warmup() {
  try {
    $("#warm").disabled = true;
    showJob(await api("/api/engine/warmup", { clip: currentClip(), preset: $("#preset").value, values: state.values }));
    toast("Warming up: the models load once, then the first frames pick the players.", 5000);
    pollJob();
  } catch (e) { $("#warm").disabled = false; toast(e.message, 8000); }
}
$("#warm").addEventListener("click", warmup);

// v0.8.7 one click: warm up on the lightest settings, stream 20 s without Unity, stop, show the time report
async function engineNow() { const d = await api("/api/job"); showJob(d.current); return d.current && d.current.kind === "engine" ? d.current : null; }
async function waitEngine(ok, seconds, what) {
  const t_end = Date.now() + seconds * 1000;
  while (Date.now() < t_end) {
    const j = await engineNow();
    if (!j || j.state !== "running") throw new Error(`the engine stopped while ${what}` + (j && j.error ? `: ${j.error}` : ""));
    const e = j.engine || {};
    if (e.error && e.error.message) throw new Error(e.error.message);
    if (ok(e)) return e;
    await pause(1000);
  }
  throw new Error(`still ${what} after ${Math.round(seconds / 60)} min`);
}
$("#testrun").addEventListener("click", async () => {
  const b = $("#testrun"), msg = $("#testrun-msg"), TEST_S = 20;
  const say = (t) => { msg.hidden = false; msg.textContent = t; };
  const noGpu = state.info.machine === "laptop-cached" && (state.input || {}).kind !== "camera";   // stored 2D instead
  const preset = noGpu ? state.info.default_settings : state.presets.some((p) => p.name === "live_test") ? "live_test" : "live_30";
  const clip = currentClip();
  b.disabled = true;
  try {
    say(`Test run 1/4: warming up on preset ${preset} (the first time on this PC can take minutes)…`);
    await api("/api/engine/warmup", { clip, preset, values: {} });
    await waitEngine((e) => ["loading", "warming"].includes(e.state), 30, "starting the warm-up").catch(() => {});
    const w = await waitEngine((e) => e.state === "ready" && e.warm, 45 * 60, "warming up");
    const players = Object.entries((w.warm || {}).players || {}).map(([p, ok]) => `${p} ${ok ? "✓" : "✗"}`).join(" ");
    say(`Test run 2/4: warm (${players}). Streaming for ${TEST_S} s…`);
    await api("/api/stream/start", { clip, preset, values: {}, force: true });
    await waitEngine((e) => e.state === "streaming", 60, "starting the stream");
    const t0 = Date.now();
    while (Date.now() - t0 < TEST_S * 1000) { await engineNow(); await pause(1000); }
    say("Test run 3/4: stopping, saving the analysis file and the report…");
    await api("/api/stream/stop", {});
    const e = await waitEngine((x) => x.state === "ready", 5 * 60, "saving the stream");
    const run = e.last_stream && e.last_stream.run_dir ? e.last_stream.run_dir.replace(/[\\/]+$/, "").split(/[\\/]/).pop() : null;
    say(`Test run 4/4: done${e.last_stream ? ` · ${e.last_stream.written} frames written` : ""}${run ? ` · ${run}` : ""}. The time report is below.`);
    await loadRuns();
    if (run) await loadTiming(run);
  } catch (err) { say("Test run failed: " + err.message + " (the full output is in runs/server_logs/)"); toast(err.message, 12000); }
  finally { b.disabled = false; pollJob(); }
});
$("#engine-off").addEventListener("click", async () => {
  try { showJob(await api("/api/engine/stop", {})); toast("Engine shut down (the models are unloaded)."); } catch (e) { toast(e.message); }
});

const ENGINE_WORDS = { starting: "Engine starting", cold: "Engine started", loading: "Loading the models",
  warming: "Warming up the players", ready: "Ready to stream", streaming: "Streaming", off: "Engine off" };

function showEngine(j) {
  const eng = j && j.kind === "engine" && j.state === "running" ? (j.engine || {}) : null;
  const pill = $("#engine"), det = $("#engine-detail");
  $("#engine-off").disabled = !eng;
  if (!eng) {
    pill.className = "pill"; pill.textContent = "Engine: off"; $("#st-stream").textContent = "";
    $("#warm").disabled = streamBlocked(); $("#warm").textContent = "1 Warm up";
    det.textContent = 'Off. Step 1 "Warm up" loads the 2D models and picks the players once; step 2 then streams from that warm state as often as you like.';
    return null;
  }
  const busy = ["starting", "cold", "loading", "warming", "streaming"].includes(eng.state);
  pill.className = "pill " + (eng.state === "ready" ? "warm" : busy ? "busy" : "");
  pill.textContent = `Engine: ${ENGINE_WORDS[eng.state] || eng.state}`;
  $("#st-stream").textContent = eng.state === "streaming" ? "streaming" : eng.warm ? "warm" : "";
  $("#warm").disabled = busy || streamBlocked();
  $("#warm").textContent = eng.warm ? "1 Warm up again" : "1 Warm up";
  const w = eng.warm;
  const bits = [`${eng.clip} · ${eng.preset}`];
  if (eng.models_s != null) bits.push(`models loaded in ${fmt(eng.models_s)} s`);
  if (w) {
    const pl = Object.entries(w.players || {}).map(([p, ok]) => `${p} ${ok ? "✓" : "✗"}`).join(" ");
    const offs = Object.entries(w.offsets || {}).map(([v, o]) => `${v.replace(/_\d+$/, "")} ${o >= 0 ? "+" : ""}${Number(o).toFixed(1)}`).join(", ");
    bits.push(`warm since ${(w.made || "").slice(11, 16)}: ${pl}, floor ${fmt((w.ankle_h_m || 0) * 100)} cm, offsets ${offs}`);
    bits.push(`${eng.streams} stream${eng.streams === 1 ? "" : "s"} from this warm-up`);
  } else bits.push(busy ? "warming up…" : "not warm: press 1 Warm up");
  if (eng.error && eng.error.message) bits.push(`last error: ${eng.error.message}`);
  det.textContent = bits.join(" · ");
  return eng;
}
$("#start").addEventListener("click", () => start(false));
$("#confirm-anyway").addEventListener("click", () => start(true));
$("#confirm-cancel").addEventListener("click", () => ($("#confirm").hidden = true));
$("#stop").addEventListener("click", async () => { try { showJob(await api("/api/stream/stop", {})); } catch (e) { toast(e.message); } });

const meta_title = (j) => (j.meta && j.meta.title) || j.title;
const mmss = (s) => (s == null || !isFinite(s) ? "–" : s < 60 ? `${Math.round(s)} s` : `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`);
const ICON = { pending: "○", done: "✓", failed: "✗", stopped: "■" };

function showJob(j) {
  const box = $("#job");
  box.className = "job " + (j ? j.state : "idle");
  const eng = showEngine(j);
  const running = eng ? ["starting", "cold", "loading", "warming", "streaming"].includes(eng.state)
    : !!(j && (j.state === "running" || j.state === "starting"));
  const streaming = eng ? eng.state === "streaming" : running;
  $("#start").disabled = eng ? eng.state !== "ready" : true; $("#stop").disabled = !streaming;
  $("#start").textContent = streaming ? "Streaming…" : "2 Start streaming to Unity";
  if (!j) {
    $("#job-title").textContent = "Idle"; $("#job-stats").textContent = ""; $("#job-folder").textContent = "";
    $("#phases").innerHTML = ""; $("#bar").className = "bar"; $("#bar-fill").style.width = "0"; $("#bar-text").textContent = "";
    $("#open-folder").hidden = true; return;
  }
  const words = { starting: "Starting", running: "Streaming", done: "Stream finished", failed: "Failed", stopped: "Stopped" };
  $("#job-title").textContent = eng ? `${ENGINE_WORDS[eng.state] || eng.state}: ${meta_title(j)}` : `${words[j.state] || j.state}: ${j.title}`;
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
  const ls = eng && eng.last_stream;
  const idleDone = eng ? (!running && ls) : j.state === "done";
  $("#bar-fill").style.width = `${idleDone ? 100 : pct}%`;
  $("#bar-text").textContent = running ? (counting ? `${pct}% · frame ${p.written} / ${p.steps}` : act ? act.label + "…" : "starting…")
    : eng ? (ls ? (ls.stopped ? `stopped · ${ls.written} frames` : "done") : (eng.warm ? "ready" : ""))
    : j.state === "done" ? "done" : words[j.state] || "";
  // speed since the first frame step (warm-up start) and time left at that speed
  const t0 = (ph.find((x) => x.id === "warmup" && x.t0) || ph.find((x) => x.id === "stream" && x.t0) || {}).t0;
  const t1 = (ph.find((x) => x.id === "stream") || {}).t1 || (running ? now : j.ended);
  const rate = t0 && p.written ? p.written / Math.max(0.1, t1 - t0) : null;
  const left = rate && p.steps && running && counting ? (p.steps - p.written) / rate : null;
  const t_first = ph.length && ph[0].t0 ? ph[0].t0 : null;           // engine: this warm-up / stream, not the engine
  const t_last = ph.length ? (ph.map((x) => x.t1).filter(Boolean).pop() || null) : null;
  const elapsed = eng && t_first ? ((running ? now : (t_last || now)) - t_first) : j.elapsed_s;
  const bits = [`elapsed ${mmss(elapsed)}`];
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
  const key = eng ? eng.state : j.state;
  if ((state.lastState === "running" && key !== "running") || (state.lastState === "streaming" && key !== "streaming")) {
    loadRuns(); if (meta.run) loadTiming(meta.run);
  }
  if (!state.lastState && !running && meta.run && j.state === "done") loadTiming(meta.run);
  state.lastState = key;
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
    ["Keeping up", !t.lag_2d_ms_start_end || t.keeps_up == null ? "–"
      : `delay ${ms((t.e2e_ms_start_end || [])[0])} → ${ms((t.e2e_ms_start_end || [])[1])}, 2D lag ${ms(t.lag_2d_ms_start_end[0])} → ${ms(t.lag_2d_ms_start_end[1])} from start to end: ` +
        (t.keeps_up ? "keeps up with the cameras (× real time also counts the start-up backlog and the fit's tail)"
          : t.keeps_up_2d ? "the 2D keeps up, but the delay grows (the body fit falls behind)" : "falls behind (the 2D lag grows)")],
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
    $("#unity-detail").textContent = (u && u.seen ? `Not in Play mode (last heartbeat ${fmt(u.age_s, 0)} s ago). Press Play in Unity.`
      : "No heartbeat yet: open the project in Unity and press Play.") + (u && u.path ? ` Heartbeat file: ${u.path}` : "");
  }
}

async function pollJob() {
  try {
    const [d, u] = await Promise.all([api("/api/job"), api("/api/unity")]);
    showJob(d.current); showUnity(u);
    const running = d.current && (d.current.state === "running" || d.current.state === "starting");   // incl. the engine
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
    [state.info, state.schema, state.clips, state.input] = await Promise.all([api("/api/info"), api("/api/schema"),
      api("/api/clips"), api("/api/input")]);
    $("#machine").textContent = state.info.machine === "laptop-cached" ? "Machine: laptop (stored 2D)" : `Machine: ${state.info.machine}`;
    $("#machine").title = `machine preset ${state.info.machine}\nUnity project ${state.info.unity_project}\nruns ${state.info.runs_dir}`;
    await camCalibStatus();
    sidebar();
    await loadSource();
    await loadPresets();
    pollJob();
    api("/api/runs").then((runs) => { const r = runs.find((x) => x.has_report); if (r && $("#timebox").hidden) loadTiming(r.name); }).catch(() => {});
  } catch (e) { toast("could not reach the server: " + e.message, 8000); }
})();
