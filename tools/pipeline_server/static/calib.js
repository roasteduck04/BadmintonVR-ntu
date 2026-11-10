// BadmintonVR live page (v0.8.6), step 2 Calibrate: court points on the frame of each camera the Source step chose
// (the clip's ticked angles, or the live cameras in use). Save coordinates = best field of view + camera pose
// (POST /calib/api/calibrate) -> data/calib/<camera>_cam.json, the previous file archived. Uses app.js's $, api, toast.
"use strict";

const Calib = (() => {
  const C = { cams: [], cam: null, court: null, names: [], pts: {}, order: [], sel: null, img: null,
    view: { s: 1, ox: 0, oy: 0 }, fit: 1, proj: {}, drag: null };
  const cv = $("#cal-canvas"), wrap = $("#cal-wrap"), ctx = cv.getContext("2d");
  const RMS_WARN = 15;

  // ---------------------------------------------------------------- cameras
  async function open() {
    try {
      if (!C.court) {
        C.court = await api("/calib/api/court-points");
        C.names = Object.keys(C.court);
      }
      C.cams = await api("/calib/api/source-cameras");
    } catch (e) { toast("calibrate: " + e.message); return; }
    const inp = state.input || {}, c = clipOf(inp.clip);
    $("#cal-sub").textContent = inp.kind === "camera" ? `The live cameras in use (${C.cams.length})`
      : `${c ? c.label : inp.clip}: the angles chosen in step 1`;
    chips();
    const keep = C.cam && C.cams.find((x) => x.view === C.cam.view);
    if (C.cams.length) select((keep || C.cams[0]).view);
    else { C.cam = null; draw(); $("#cal-pts tbody").innerHTML = ""; $("#cal-next").textContent = "No cameras: choose them in step 1."; }
    $("#sync-card").hidden = inp.kind === "camera";
    if (inp.kind !== "camera") loadSync();
  }

  function chips() {
    $("#cal-cams").innerHTML = C.cams.map((c) => `<button class="cam${C.cam && C.cam.view === c.view ? " on" : ""}" data-view="${esc(c.view)}">
      ${esc(pretty(c.view))} ${calChip(c.rms, c.calibrated)}</button>`).join("");
  }
  $("#cal-cams").addEventListener("click", (ev) => {
    const b = ev.target.closest(".cam");
    if (b) select(b.dataset.view);
  });

  function select(view) {
    const cam = C.cams.find((x) => x.view === view);
    if (!cam) return;
    C.cam = cam;
    C.pts = Object.fromEntries(Object.entries(cam.saved_points || {}).map(([n, p]) => [n, [p[0], p[1]]]));
    C.order = Object.keys(C.pts);
    C.proj = Object.fromEntries(Object.entries(cam.saved_proj || {}).map(([n, p]) => [n, p]));
    C.sel = nextName();
    readout(cam.saved_fov, cam.rms);
    $("#cal-path").textContent = cam.calibrated ? `Saved ${cam.made || ""} in data/calib/${cam.view}_cam.json` : `Not calibrated yet: saving writes data/calib/${cam.view}_cam.json`;
    chips(); loadFrame(); listPoints(); court();
  }

  function loadFrame(bust) {
    if (!C.cam) return;
    $("#cal-loading").hidden = false;
    const img = new Image(), view = C.cam.view;
    C.loading = img;                                  // only the newest request may show (a slow first grab of
    const mine = () => C.loading === img && C.cam && C.cam.view === view;      // another camera arrives late)
    img.onload = () => { if (!mine()) return; C.img = img; $("#cal-loading").hidden = true; fitView(); };
    img.onerror = () => { if (mine()) $("#cal-loading").textContent = "No frame (video / camera not reachable)."; };
    img.src = `/calib/frames/${encodeURIComponent(C.cam.view)}.jpg?t=${bust || Date.now()}`;
    $("#cal-loading").textContent = "Loading the frame…";
  }

  // ---------------------------------------------------------------- view
  function resize() {
    const r = wrap.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    cv.width = Math.max(1, Math.round(r.width * dpr)); cv.height = Math.max(1, Math.round(r.height * dpr));
  }
  function fitView() {
    resize();
    if (!C.img) { draw(); return; }
    const r = wrap.getBoundingClientRect();
    const s = Math.min(r.width / C.img.width, r.height / C.img.height);
    C.fit = s;
    C.view = { s, ox: (r.width - C.img.width * s) / 2, oy: (r.height - C.img.height * s) / 2 };
    draw();
  }
  new ResizeObserver(() => { if (!$("#step-calibrate").hidden) fitView(); }).observe(wrap);
  const toImg = (mx, my) => [(mx - C.view.ox) / C.view.s, (my - C.view.oy) / C.view.s];
  const toScr = (x, y) => [x * C.view.s + C.view.ox, y * C.view.s + C.view.oy];

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, cv.width, cv.height);
    if (!C.img || !C.cam) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = C.view.s < 2;
    ctx.drawImage(C.img, C.view.ox, C.view.oy, C.img.width * C.view.s, C.img.height * C.view.s);
    ctx.lineWidth = 2; ctx.font = "600 12px system-ui, sans-serif";
    for (const [n, p] of Object.entries(C.proj)) {          // where the saved calibration puts each clicked point
      if (!C.pts[n]) continue;
      const [ax, ay] = toScr(p[0], p[1]), [bx, by] = toScr(C.pts[n][0], C.pts[n][1]);
      ctx.strokeStyle = p[2] > RMS_WARN ? "#ff5a4f" : "rgba(120, 220, 255, .9)";
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
      ctx.beginPath(); ctx.arc(ax, ay, 5, 0, Math.PI * 2); ctx.stroke();
    }
    C.order.forEach((n, i) => {
      const [x, y] = toScr(C.pts[n][0], C.pts[n][1]);
      const bad = C.proj[n] && C.proj[n][2] > RMS_WARN;
      ctx.strokeStyle = bad ? "#ff5a4f" : "#ffd23f"; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x - 3, y); ctx.lineTo(x + 3, y); ctx.moveTo(x, y - 3); ctx.lineTo(x, y + 3); ctx.stroke();
      const label = `${i + 1} ${n}`;
      ctx.lineWidth = 3; ctx.strokeStyle = "rgba(0,0,0,.75)"; ctx.strokeText(label, x + 10, y - 9);
      ctx.fillStyle = "#fff"; ctx.fillText(label, x + 10, y - 9);
    });
  }

  // ---------------------------------------------------------------- clicks, zoom, pan
  const mpos = (ev) => { const r = cv.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top]; };
  cv.addEventListener("wheel", (ev) => {
    if (!C.img) return;
    ev.preventDefault();
    const [mx, my] = mpos(ev);
    const s0 = C.view.s, s1 = Math.min(C.fit * 40, Math.max(C.fit * 0.5, s0 * Math.exp(-ev.deltaY * 0.0015)));
    C.view.ox = mx - (mx - C.view.ox) * (s1 / s0); C.view.oy = my - (my - C.view.oy) * (s1 / s0); C.view.s = s1;
    draw();
  }, { passive: false });
  cv.addEventListener("mousedown", (ev) => {
    const [mx, my] = mpos(ev);
    C.drag = { x: mx, y: my, ox: C.view.ox, oy: C.view.oy, moved: false, button: ev.button };
  });
  window.addEventListener("mousemove", (ev) => {
    if (!C.drag) return;
    const [mx, my] = mpos(ev);
    if (Math.hypot(mx - C.drag.x, my - C.drag.y) > 4) { C.drag.moved = true; wrap.classList.add("dragging"); }
    if (C.drag.moved) { C.view.ox = C.drag.ox + mx - C.drag.x; C.view.oy = C.drag.oy + my - C.drag.y; draw(); }
  });
  window.addEventListener("mouseup", (ev) => {
    const d = C.drag; C.drag = null; wrap.classList.remove("dragging");
    if (!d || d.moved || !C.img || ev.target !== cv) return;
    const [ix, iy] = toImg(...mpos(ev));
    if (d.button === 2) { removeNear(ix, iy); return; }
    if (d.button !== 0 || ix < 0 || iy < 0 || ix > C.img.width || iy > C.img.height) return;
    if (!C.sel) { toast("All points are placed: pick one on the court diagram to move it."); return; }
    place(C.sel, ix, iy);
  });
  cv.addEventListener("contextmenu", (ev) => ev.preventDefault());

  function place(name, x, y) {
    C.pts[name] = [x, y];
    C.order = C.order.filter((n) => n !== name).concat(name);
    delete C.proj[name];                                     // its saved reprojection no longer applies
    C.sel = nextName();
    listPoints(); court(); draw();
  }
  function removeNear(ix, iy) {
    let best = null, bd = 14 / C.view.s;
    for (const n of C.order) { const d = Math.hypot(C.pts[n][0] - ix, C.pts[n][1] - iy); if (d < bd) { bd = d; best = n; } }
    if (best) remove(best);
  }
  function remove(name) {
    delete C.pts[name]; delete C.proj[name];
    C.order = C.order.filter((n) => n !== name); C.sel = name;
    listPoints(); court(); draw();
  }
  const nextName = () => C.names.find((n) => !C.pts[n]) || null;

  function listPoints() {
    $("#cal-pts tbody").innerHTML = C.order.map((n) => {
      const p = C.pts[n], e = C.proj[n] ? C.proj[n][2] : null;
      return `<tr class="${n === C.sel ? "sel" : ""}"><td>${n}</td><td>${Math.round(p[0])}, ${Math.round(p[1])}</td>
        <td class="${e == null ? "" : e > RMS_WARN ? "badc" : "good"}">${e == null ? "–" : e.toFixed(1)}</td>
        <td class="x" data-rm="${n}" title="remove">×</td></tr>`;
    }).join("") || `<tr><td colspan="4" class="muted">No points yet.</td></tr>`;
    $("#cal-next").innerHTML = C.sel ? `Next: <b>${C.sel}</b> (${C.order.length} placed)` : `${C.order.length} points placed`;
  }
  $("#cal-pts").addEventListener("click", (ev) => { const n = ev.target.dataset && ev.target.dataset.rm; if (n) remove(n); });

  // ---------------------------------------------------------------- court diagram
  function court() {
    if (!C.court) return;
    const k = Math.min(120 / 6.1, 250 / 13.4), X = (x) => 75 + x * k, Y = (z) => 140 - z * k;
    const line = (x1, z1, x2, z2, cls = "ln") => `<line class="${cls}" x1="${X(x1)}" y1="${Y(z1)}" x2="${X(x2)}" y2="${Y(z2)}"/>`;
    let h = `<rect class="surf" x="${X(-3.05)}" y="${Y(6.7)}" width="${6.1 * k}" height="${13.4 * k}"/>`;
    h += line(-2.59, -6.7, -2.59, 6.7) + line(2.59, -6.7, 2.59, 6.7);
    h += line(-3.05, 1.98, 3.05, 1.98) + line(-3.05, -1.98, 3.05, -1.98);
    h += line(-3.05, 5.94, 3.05, 5.94) + line(-3.05, -5.94, 3.05, -5.94);
    h += line(0, 1.98, 0, 6.7) + line(0, -1.98, 0, -6.7) + line(-3.35, 0, 3.35, 0, "net");
    for (const [n, [x, z]] of Object.entries(C.court)) {
      const cls = ["pt", C.pts[n] ? "done" : "", n === C.sel ? "sel" : ""].join(" ");
      h += `<circle class="${cls}" data-pt="${n}" cx="${X(x)}" cy="${Y(z)}" r="${n === C.sel ? 5.5 : 4.5}"><title>${n}</title></circle>`;
    }
    h += `<text class="cap" x="75" y="${Y(-6.7) + 11}" text-anchor="middle">near side</text>`;
    $("#cal-court").innerHTML = h;
  }
  $("#cal-court").addEventListener("click", (ev) => {
    const n = ev.target.dataset && ev.target.dataset.pt;
    if (!n) return;
    C.sel = n; listPoints(); court();
  });

  // ---------------------------------------------------------------- buttons
  function readout(fov, rms) {
    $("#cal-fov").textContent = fov == null ? "found on save" : `${Number(fov).toFixed(2)}°`;
    $("#cal-rms").textContent = rms == null ? "–" : `${Number(rms).toFixed(2)} px`;
    $("#cal-rms").className = rms == null ? "" : rms > RMS_WARN ? "badc" : "good";
  }
  function undo() { const n = C.order[C.order.length - 1]; if (n) remove(n); }
  $("#cal-undo").addEventListener("click", undo);
  $("#cal-clear").addEventListener("click", () => {
    if (!C.order.length || !confirm("Remove every point of this camera? (The saved file stays until you save.)")) return;
    C.pts = {}; C.order = []; C.proj = {}; C.sel = nextName(); listPoints(); court(); draw();
  });
  $("#cal-fit").addEventListener("click", fitView);
  $("#cal-frame").addEventListener("click", async () => {
    if (!C.cam) return;
    let body = { view: C.cam.view };
    if (C.cam.input !== "camera") {
      const f = prompt("Show which frame of this video? (frame number; 60 per second)", "300");
      if (f === null) return;
      body.frame = Number(f);
    }
    try { await api("/calib/api/re-extract", body); loadFrame(); }
    catch (e) { toast(e.message, 6000); }
  });

  async function save() {
    if (!C.cam) return;
    if (C.order.length < 4) { toast("Place at least 4 court points (6+ spread over the court is better)."); return; }
    const btn = $("#cal-save"); btn.disabled = true; btn.textContent = "Finding the field of view…";
    try {
      const r = await api("/calib/api/calibrate", { view: C.cam.view, fov_on: "long",
        points: C.order.map((n) => ({ name: n, px: C.pts[n][0], py: C.pts[n][1] })) });
      C.proj = Object.fromEntries(r.points.map((p) => [p.name, [p.proj[0], p.proj[1], p.err_px]]));
      readout(r.fov_deg, r.rms_px);
      $("#cal-path").textContent = `Saved in ${r.path}${r.archived ? ` (old one in ${r.archived})` : ""}`;
      toast(r.warning ? `Saved, but RMS ${r.rms_px.toFixed(1)} px is above ${RMS_WARN}: the red line shows the worst click (${r.worst.name}). Redo it and save again.`
        : `Saved: field of view ${r.fov_deg.toFixed(2)}°, RMS ${r.rms_px.toFixed(2)} px.`, r.warning ? 9000 : 5000);
      const cam = C.cams.find((x) => x.view === C.cam.view);
      Object.assign(cam, { calibrated: true, rms: r.rms_px, saved_fov: r.fov_deg,
        saved_points: Object.fromEntries(C.order.map((n) => [n, C.pts[n]])), saved_proj: { ...C.proj } });
      chips(); listPoints(); draw();
      state.clips = await api("/api/clips"); await camCalibStatus(); sidebar();
    } catch (e) { toast(e.message, 8000); }
    finally { btn.disabled = false; btn.textContent = "Save coordinates (finds FOV)"; }
  }
  $("#cal-save").addEventListener("click", save);

  document.addEventListener("keydown", (ev) => {
    if ($("#step-calibrate").hidden || ["INPUT", "SELECT", "TEXTAREA"].includes(ev.target.tagName) || ev.ctrlKey || ev.metaKey) return;
    if (ev.key === "z" || ev.key === "Z") { undo(); ev.preventDefault(); }
    if (ev.key === "s" || ev.key === "S") { if (!$("#cal-save").disabled) save(); ev.preventDefault(); }
    if (ev.key === "Escape") { C.sel = null; listPoints(); court(); }
  });

  // ---------------------------------------------------------------- camera sync (recorded videos)
  async function loadSync() {
    const clip = currentClip(), c = clipOf(clip);
    try {
      const rows = (await api("/api/sync")).filter((r) => r.clip === clip);
      $("#sync tbody").innerHTML = rows.map((r) => {
        const views = c ? chosenViews(c) : [], done = views.filter((v) => c && c.calibrated[v]).length;
        const s = r.sync;
        const offs = s ? Object.entries(s.offsets).filter(([v]) => views.includes(v))
          .map(([v, o]) => `${pretty(v)} ${o >= 0 ? "+" : ""}${o}`).join(", ") : "–";
        return `<tr><td class="name">${clip}</td>
          <td class="${done === views.length ? "good" : "badc"}">${done} / ${views.length}</td>
          <td>${s ? `${esc(s.made || "")} · from ${esc(s.source || "?")}` : '<span class="muted">none</span>'}</td>
          <td>${offs}</td>
          <td><button class="btn small" data-sync="${clip}" ${r.has_build ? "" : 'disabled title="no offline build of this clip"'}>${s ? "Remake" : "Make"} from the offline build</button></td></tr>`;
      }).join("");
    } catch (e) { toast("sync: " + e.message); }
  }
  $("#sync").addEventListener("click", async (ev) => {
    const c = ev.target.dataset && ev.target.dataset.sync;
    if (!c) return;
    try {
      const d = await api("/api/sync", { clip: c });
      toast(`Sync file for ${c} written${d.archived ? " (old one archived)" : ""}.`);
      state.clips = await api("/api/clips"); loadSync();
    } catch (e) { toast(e.message, 7000); }
  });

  return { open };
})();
