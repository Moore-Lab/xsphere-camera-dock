/* xsphere camera dock — camera page.
 *
 * All coordinate mapping follows docs/DESIGN.md §4 exactly:
 *  - server-reported image_size (Iw, Ih) is authoritative; img.naturalWidth is
 *    only a staleness GATE (the MJPEG can still be showing old-geometry frames);
 *  - mapping inputs are captured at pointerdown and the drag is cancelled if
 *    they changed by pointerup;
 *  - the echoed APPLIED ROI is adopted (hardware snaps requests);
 *  - the zoom stack pushes the PREVIOUS applied ROI only after a 2xx;
 *  - after every successful ROI change the stream <img> reconnects with a fresh
 *    epoch (an <img> never auto-reconnects a stalled MJPEG response).
 * Everything is in CSS pixels — devicePixelRatio is irrelevant by design.
 */
"use strict";

const BASE = location.pathname.replace(/\/$/, "");
const NAME = BASE.split("/").pop();
const $ = id => document.getElementById(id);

document.getElementById("rootlink").href =
  BASE.replace(/\/cam\/[^/]+$/, "") + "/";
document.title = NAME;
$("title").textContent = NAME;

// ------------------------------------------------------------------ state --
let R = {                   // mirror of last server echo
  roi: null,                // [x, y, w, h] sensor coords
  imageSize: null,          // [Iw, Ih]
  sensor: null,             // [Sw, Sh]
  binning: [1, 1],
  hasRoi: false,
};
let caps = null;            // last capabilities document
let capsRev = 0;            // bumped on every capabilities adoption
const zoomStack = [];       // previous applied ROIs (undo)
let epoch = 0;              // stream reconnect counter
let recState = "idle";
let tlOn = false, tlTimer = null;

// Measuring is done by the SHARED layer (measure.js) — the same code the
// offline analysis page uses. This page only supplies the live-specific bits:
// the sensor-coordinate transform, the ROI tool, and server-backed calibration.
let M = null;               // MeasureLayer instance
let tool = "roi";           // roi (ours) | line | circle | rect (the layer's)
let umPerPx = null;         // active calibration scale (null = pixels only)
let calDoc = { active: null, entries: {} };
const ANNOT_KEY = "xsphere.annots." + NAME;

// ------------------------------------------------------------------ utils --
async function post(url) {
  const r = await fetch(BASE + url, { method: "POST" });
  let j = {};
  try { j = await r.json(); } catch (e) { }
  return { ok: r.ok, status: r.status, j };
}
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.style.display = "block";
  clearTimeout(t._h);
  t._h = setTimeout(() => { t.style.display = "none"; }, 3500);
}
function stat(msg) { $("stat").textContent = msg; }
function roisEqual(a, b) {
  return a && b && a.length === 4 && a.every((v, i) => v === b[i]);
}
function resetStream() {
  epoch += 1;
  $("stream").src = `${BASE}/stream?e=${epoch}&hist=${$("hist").checked ? 1 : 0}`;
}
$("hist").onchange = resetStream;

// ------------------------------------------------- generic control panel --
// One row per control from the capabilities document: a 0..1000 slider posting
// normalized values, a native-value box, unit label. No per-control hardcoding.
const debounces = {};       // name -> timer
const rowState = {};        // name -> {sentRev}

function displayValue(spec, v) {
  if (v == null) return "—";
  const scale = spec.display && spec.display.unit_scale;
  const unit = (spec.display && spec.display.unit_display) || spec.units;
  const shown = scale ? v * scale : v;
  const digits = (spec.kind === "int" && !scale) ? 0 : 2;
  let txt = `${shown.toFixed(digits)} ${unit}`;
  if (spec.display && spec.display.db_hi > 0 && spec.hi > spec.lo) {
    const db = spec.display.db_lo +
      (v - spec.lo) / (spec.hi - spec.lo) * (spec.display.db_hi - spec.display.db_lo);
    txt += ` (~${db.toFixed(1)} dB)`;
  }
  return txt;
}

function buildControls() {
  const host = $("controls");
  host.innerHTML = "";
  for (const [name, spec] of Object.entries(caps.controls)) {
    const row = document.createElement("div");
    row.className = "row ctlrow";
    row.innerHTML = `
      <label>${(spec.display && spec.display.label) || name}
        <span class="val" id="ctl-${name}-v"></span></label>
      <input type="range" id="ctl-${name}-s" min="0" max="1000">
      <input type="text" id="ctl-${name}-t" title="native value (${spec.units})">`;
    host.appendChild(row);
    bindControl(name, spec);
  }
}

function bindControl(name, spec) {
  const s = $(`ctl-${name}-s`), t = $(`ctl-${name}-t`), v = $(`ctl-${name}-v`);
  const show = val => {
    v.textContent = displayValue(spec, val);
    t.value = val == null ? "" : (spec.kind === "int" ? Math.round(val) : val.toFixed(3));
  };
  s.value = Math.round((spec.normalized || 0) * 1000);
  show(spec.value);
  rowState[name] = rowState[name] || {};

  const send = async (url) => {
    const sentRev = capsRev;
    const { ok, status, j } = await post(url);
    if (sentRev !== capsRev) return;          // superseded by a caps refresh
    if (!ok) { toast(`${name}: ${j.detail || status}`); return; }
    show(j.value);
    s.value = Math.round((j.normalized || 0) * 1000);
    if (j.warning) toast(j.warning);
    if (j.side_effects && Object.keys(j.side_effects).length) loadCaps();
  };
  s.oninput = () => {
    clearTimeout(debounces[name]);
    debounces[name] = setTimeout(() =>
      send(`/control/${name}?value=${s.value / 1000}&normalized=true`), 80);
  };
  const typed = () => {
    let val = parseFloat(t.value);
    if (!isFinite(val)) return;
    const scale = spec.display && spec.display.unit_scale;
    // typed values are native units (the box shows native)
    send(`/control/${name}?value=${val}`);
  };
  t.onchange = typed;
}

function cancelDebounces() {
  for (const k of Object.keys(debounces)) clearTimeout(debounces[k]);
}

// ------------------------------------------------------------ capabilities --
function adoptCaps(c) {
  caps = c;
  capsRev += 1;
  if (M) M.epoch = capsRev;   // discards any drag started under the old geometry
  R.hasRoi = !!c.has_roi;
  R.imageSize = c.image_size || null;
  R.sensor = c.sensor_size || null;
  R.binning = c.binning || [1, 1];
  if (c.roi) R.roi = c.roi;
  buildControls();
  drawRoiFields();
  $("roi-row").style.display = R.hasRoi ? "" : "none";
  redraw();                  // annotations follow the new geometry
}

async function loadCaps() {
  try {
    const r = await fetch(BASE + "/capabilities");
    if (!r.ok) return;
    adoptCaps(await r.json());
  } catch (e) { }
}

function drawRoiFields() {
  if (!R.roi) return;
  $("roi-x").value = R.roi[0]; $("roi-y").value = R.roi[1];
  $("roi-w").value = R.roi[2]; $("roi-h").value = R.roi[3];
  $("roi-eff").textContent = R.imageSize ? `image ${R.imageSize[0]}×${R.imageSize[1]}` : "";
}

// ------------------------------------------------------- coordinate maps ---
// The <img> shows the current ROI at some displayed size. Annotations are kept
// in sensor pixels, so both directions go through the ROI origin and binning.
function sensorToDisplay(sx, sy) {
  const rect = contentRect();
  const [Rx, Ry] = R.roi, [bx, by] = R.binning, [Iw, Ih] = R.imageSize;
  return { u: (sx - Rx) / bx * (rect.w / Iw), v: (sy - Ry) / by * (rect.h / Ih) };
}
function displayToSensor(u, v) {
  const rect = contentRect();
  const [Rx, Ry] = R.roi, [bx, by] = R.binning, [Iw, Ih] = R.imageSize;
  return { x: Rx + (u * Iw / rect.w) * bx, y: Ry + (v * Ih / rect.h) * by };
}
function mappable() { return !!(R.roi && R.imageSize && R.binning); }

// ---------------------------------------------------------------- ROI ops --
// Every successful change adopts the server echo; pushes previous ROI (if the
// caller asks and it actually changed); resets the stream connection; cancels
// stale slider debounces; re-renders capabilities LAST.
async function applyRoi(url, { push } = { push: true }) {
  const prev = R.roi ? [...R.roi] : null;
  const { ok, status, j } = await post(url);
  if (!ok) { toast(`ROI: ${j.detail || status}`); return false; }
  cancelDebounces();
  if (push && prev && j.roi && !roisEqual(prev, j.roi)) zoomStack.push(prev);
  adoptCaps(j.capabilities);
  resetStream();
  return true;
}

$("roi-apply").onclick = () => {
  const v = ["roi-x", "roi-y", "roi-w", "roi-h"].map(id => parseInt($(id).value, 10));
  if (v.some(x => !isFinite(x))) { toast("ROI: enter x, y, w, h"); return; }
  applyRoi(`/roi?x=${v[0]}&y=${v[1]}&w=${v[2]}&h=${v[3]}`);
};
$("roi-full").onclick = async () => {
  if (await applyRoi("/roi/reset", { push: false })) zoomStack.length = 0;
};

// Right-click: uniform undo of the last ROI change; empty stack -> full frame
// (no-op when already there). A pop never pushes.
$("streambox").addEventListener("contextmenu", async ev => {
  ev.preventDefault();
  if (!R.hasRoi) return;
  if (zoomStack.length) {
    const target = zoomStack[zoomStack.length - 1];
    const okd = await applyRoi(
      `/roi?x=${target[0]}&y=${target[1]}&w=${target[2]}&h=${target[3]}`,
      { push: false });
    if (okd) zoomStack.pop();
  } else if (R.sensor && R.roi &&
             !roisEqual(R.roi, [0, 0, R.sensor[0], R.sensor[1]])) {
    applyRoi("/roi/reset", { push: false });
  }
});

// ------------------------------------------------------- measurement layer --
// The drawing tools, shapes, labels, table and persistence all come from
// measure.js. What is specific to the LIVE page is the transform below: the
// shared layer stores annotations in "native" pixels, which here means SENSOR
// pixels, so a measurement stays pinned to the scene across ROI crops.
//
// NOTE[live->offline]: this transform IS the whole online/offline bridge.
// Offline hosts pass nothing and get MeasureLayer.defaultTransform (file pixels
// -> displayed box). Keep both sides in sync if the mapping ever changes.
const liveTransform = {
  ready: () => mappable(),
  imageSize: () => R.imageSize,
  toDisplay: (x, y) => sensorToDisplay(x, y),
  toNative: (u, v) => displayToSensor(u, v),
};

function renderMeasures() {
  M.renderList();
  const n = M.annots.length;
  $("measure-count").textContent = n ? `${n}` : "";
  $("measure-empty").style.display = n ? "none" : "";
  renderCalSources();
}

function redraw() { if (M) M.redraw(); }

// ------------------------------------------------------------------ tools --
const ROI_HINT =
  "drag: crop the sensor to that region (raises frame rate) · right-click: step back out";
function setTool(t) {
  tool = t;
  M.setTool(t);                        // highlights the button, incl. our "roi"
  sbox.classList.toggle("draw", t !== "roi");
  $("hint").textContent = (t === "roi" ? ROI_HINT
    : M.hint() + " · right-click: step back out of a zoom");
}

// ------------------------------------------------- pointer plumbing --------
// The shared layer handles its own drawing tools; these handlers only run for
// the dock's own ROI tool.
const img = $("stream"), box = $("dragbox"), sbox = $("streambox"), svg = $("overlay");
img.addEventListener("dragstart", ev => ev.preventDefault());
let drag = null;            // ROI drag: {x0, y0, snap:{rect, Iw, Ih, roi, bin, rev}}

function contentRect() {
  // Displayed content area EXCLUDING the css border (mapping spec).
  const r = img.getBoundingClientRect();
  const bl = img.clientLeft, bt = img.clientTop;
  return { left: r.left + bl, top: r.top + bt,
           w: r.width - 2 * bl, h: r.height - 2 * bt };
}

sbox.addEventListener("pointerdown", ev => {
  if (ev.button !== 0 || tool !== "roi" || !R.hasRoi || !R.roi || !R.imageSize) return;
  const [Iw, Ih] = R.imageSize;
  // Staleness gate: the <img> must actually be showing current-geometry frames.
  if (img.naturalWidth !== Iw || img.naturalHeight !== Ih) {
    toast("stream still updating — try again"); return;
  }
  drag = { x0: ev.clientX, y0: ev.clientY,
           snap: { rect: contentRect(), Iw, Ih, roi: [...R.roi],
                   bin: [...R.binning], rev: capsRev } };
  try { sbox.setPointerCapture(ev.pointerId); } catch (e) { }
  ev.preventDefault();
});

sbox.addEventListener("pointermove", ev => {
  if (!drag) return;
  const { rect } = drag.snap;
  const x = Math.min(Math.max(ev.clientX, rect.left), rect.left + rect.w);
  const y = Math.min(Math.max(ev.clientY, rect.top), rect.top + rect.h);
  const bx = sbox.getBoundingClientRect();
  box.style.display = "block";
  box.style.left = (Math.min(drag.x0, x) - bx.left) + "px";
  box.style.top = (Math.min(drag.y0, y) - bx.top) + "px";
  box.style.width = Math.abs(x - drag.x0) + "px";
  box.style.height = Math.abs(y - drag.y0) + "px";
});

sbox.addEventListener("pointerup", ev => {
  if (!drag) return;
  box.style.display = "none";
  const d = drag; drag = null;
  const { rect, Iw, Ih, roi, bin, rev } = d.snap;
  // Cancel if mapping inputs changed mid-drag (geometry moved under us).
  if (rev !== capsRev) return;
  const x1 = Math.min(Math.max(ev.clientX, rect.left), rect.left + rect.w);
  const y1 = Math.min(Math.max(ev.clientY, rect.top), rect.top + rect.h);
  if (Math.abs(x1 - d.x0) < 8 || Math.abs(y1 - d.y0) < 8) return;  // accidental click

  // display px -> image px (floor/ceil corners, inclusive), spec transform
  const u0 = Math.min(d.x0, x1) - rect.left, u1 = Math.max(d.x0, x1) - rect.left;
  const v0 = Math.min(d.y0, y1) - rect.top, v1 = Math.max(d.y0, y1) - rect.top;
  const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
  const ix0 = clamp(Math.floor(u0 * Iw / rect.w), 0, Iw - 1);
  const ix1 = clamp(Math.ceil(u1 * Iw / rect.w) - 1, ix0, Iw - 1);
  const iy0 = clamp(Math.floor(v0 * Ih / rect.h), 0, Ih - 1);
  const iy1 = clamp(Math.ceil(v1 * Ih / rect.h) - 1, iy0, Ih - 1);
  // image px -> full-sensor px (offset by current ROI, scale by binning)
  const x = roi[0] + ix0 * bin[0];
  const y = roi[1] + iy0 * bin[1];
  const w = (ix1 - ix0 + 1) * bin[0];
  const h = (iy1 - iy0 + 1) * bin[1];
  applyRoi(`/roi?x=${x}&y=${y}&w=${w}&h=${h}`);
});

// ----------------------------------------------------------- calibration --
const num = MeasureLayer.num;

function applyCalDoc(doc) {
  calDoc = doc || { active: null, entries: {} };
  umPerPx = calDoc.entry ? Number(calDoc.entry.um_per_px) : null;
  // The shared layer owns the badge, the label formatting and the redraw.
  M.setScale(umPerPx, calDoc.entry ? (calDoc.entry.label || calDoc.active) : "");
  renderCal();
  renderMeasures();
}
function renderCal() {
  const sel = $("cal-list");
  const names = Object.keys(calDoc.entries || {});
  const keep = sel.value;
  sel.innerHTML = names.length
    ? names.map(n => `<option value="${n}"${n === calDoc.active ? " selected" : ""}>` +
        `${n} — ${num(calDoc.entries[n].um_per_px)} µm/px</option>`).join("")
    : `<option value="">(none saved)</option>`;
  if (keep && names.includes(keep)) sel.value = keep;
}
function renderCalSources() {
  const sel = $("cal-src");
  const lines = M.lines;
  const keep = sel.value;
  sel.innerHTML = lines.length
    ? lines.map(a => `<option value="${a.id}">line ` +
        `${num(MeasureLayer.lineLength(a))} px</option>`).join("")
    : `<option value="">(draw a line first)</option>`;
  if (keep) sel.value = keep;
}
async function calPost(url) {
  const { ok, status, j } = await post(url);
  if (!ok) { $("cal-stat").textContent = j.detail || `error ${status}`; return false; }
  applyCalDoc(j);
  return true;
}
$("cal-load").onclick = async () => {
  const n = $("cal-list").value;
  if (!n) return;
  if (await calPost(`/calibration/load?cal=${encodeURIComponent(n)}`))
    $("cal-stat").textContent = `loaded ${n}`;
};
$("cal-clear").onclick = async () => {
  if (await calPost("/calibration/clear"))
    $("cal-stat").textContent = "cleared — showing pixels only";
};
$("cal-del").onclick = async () => {
  const n = $("cal-list").value;
  if (!n) return;
  if (await calPost(`/calibration/delete?cal=${encodeURIComponent(n)}`))
    $("cal-stat").textContent = `deleted ${n}`;
};
$("cal-save").onclick = async () => {
  const n = ($("cal-name").value || "default").trim();
  const v = parseFloat($("cal-scale").value);
  if (!isFinite(v) || v <= 0) { $("cal-stat").textContent = "enter a positive µm/px"; return; }
  if (await calPost(`/calibration/save?cal=${encodeURIComponent(n)}&um_per_px=${v}`))
    $("cal-stat").textContent = `saved ${n} and made it active`;
};
$("cal-derive").onclick = async () => {
  const a = M.byId(parseInt($("cal-src").value, 10));
  const known = parseFloat($("cal-known").value);
  if (!a) { $("cal-stat").textContent = "draw a line across a known feature first"; return; }
  if (!isFinite(known) || known <= 0) { $("cal-stat").textContent = "enter the true length in µm"; return; }
  const px = MeasureLayer.lineLength(a);
  const scale = known / px;
  $("cal-scale").value = scale.toPrecision(6);
  const n = ($("cal-name").value || "default").trim();
  if (await calPost(`/calibration/save?cal=${encodeURIComponent(n)}&um_per_px=${scale}` +
                    `&notes=${encodeURIComponent(`${num(px)} px = ${known} µm`)}`))
    $("cal-stat").textContent = `${num(px)} px = ${known} µm → ${num(scale)} µm/px (saved as ${n})`;
};
async function loadCal() {
  try { applyCalDoc(await (await fetch(BASE + "/calibration")).json()); }
  catch (e) { }
}

// ------------------------------------------------------------- actions -----
$("autoexp").onclick = async () => {
  stat("auto-exposing…");
  const { ok, j } = await post("/auto_exposure");
  stat(ok ? `exposure → ${(j.exposure / 1000).toFixed(2)} ms` : `auto-exposure: ${j.detail}`);
  loadCaps();
};
$("snap").onclick = async () => {
  const { ok, j } = await post(`/snapshot/save?fmt=${$("snap-fmt").value}`);
  stat(ok ? `snapshot → ${j.path}` : `snapshot: ${j.detail}`);
};

// ------------------------------------------------------------- recording ---
function setRecUI(state) {
  recState = state;
  const b = $("rec");
  b.textContent = state === "recording" ? "■ stop" :
                  state === "encoding" ? "encoding…" : "● record";
  b.classList.toggle("on", state === "recording");
  b.disabled = state === "encoding";
}
$("rec").onclick = async () => {
  if (recState === "idle") {
    const sec = parseFloat($("rec-sec").value) || 0;
    const fr = parseInt($("rec-frames").value, 10) || 0;
    const { ok, j } = await post(`/record/start?max_seconds=${sec}&max_frames=${fr}`);
    if (!ok) { toast(`record: ${j.detail}`); return; }
    setRecUI("recording");
    stat(`recording → ${j.path}`);
  } else if (recState === "recording") {
    const { ok, j } = await post("/record/stop");
    if (!ok) { toast(`record: ${j.detail}`); return; }
    setRecUI("encoding");
    stat("encoding…");
  }
};
async function pollRecord() {
  try {
    const j = await (await fetch(BASE + "/record/status")).json();
    if (j.rec_state !== recState) setRecUI(j.rec_state);
    if (j.rec_state === "recording") {
      $("rec-info").textContent = `${j.captured} frames · ${j.elapsed_s}s` +
        (j.limit_reached ? " · limit reached" : "");
      if (j.limit_reached) { await post("/record/stop"); setRecUI("encoding"); }
    } else if (j.rec_state === "encoding") {
      $("rec-info").textContent = "encoding…";
    } else if (j.stats && j.stats.path && $("rec-info").textContent) {
      const s = j.stats;
      $("rec-info").textContent = "";
      stat(`saved ${s.path} (${s.encoded} frames @ ${s.capture_fps} fps, ` +
           `dropped ${s.dropped}${s.sidecar ? ", sidecars ok" : ""})`);
    }
  } catch (e) { }
  setTimeout(pollRecord, 1000);
}

// ------------------------------------------------------------- timelapse ---
function fmtDur(s) {
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = s % 60;
  const p = []; if (h) p.push(h + "h"); if (m) p.push(m + "m"); if (!h) p.push(ss + "s");
  return p.join(" ");
}
function setTl(on) {
  tlOn = on;
  $("tl-btn").innerHTML = on ? "&#9632; stop timelapse" : "&#9654; start timelapse";
  $("tl-btn").style.background = on ? "#a33" : "";
}
function tlLine(j) {
  let t = `${j.taken} stills`;
  if (j.running) {
    t += ` · next in ${Math.max(0, Math.round(j.next_in_s))}s`;
    if (j.remaining_s != null) t += ` · ${fmtDur(j.remaining_s)} left`;
  }
  t += ` → ${j.dir}`;
  if (j.missed) t += ` · ${j.missed} missed`;
  if (j.stale) t += ` · ${j.stale} stale (camera not delivering?)`;
  if (j.error) t += ` · ERROR: ${j.error}`;
  return t;
}
function schedTl() { clearTimeout(tlTimer); tlTimer = setTimeout(pollTl, 2000); }
$("tl-btn").onclick = async () => {
  const btn = $("tl-btn");
  if (btn.disabled) return;
  btn.disabled = true;
  try {
    if (!tlOn) {
      const i = parseFloat($("tl-int").value), h = parseFloat($("tl-hrs").value);
      if (!isFinite(i) || i < 1) { $("tl-stat").textContent = "invalid interval (min 1 s)"; return; }
      if (!isFinite(h) || h < 0) { $("tl-stat").textContent = "invalid hours (0 = until stopped)"; return; }
      const q = `interval=${i}&hours=${h}&dir=${encodeURIComponent($("tl-dir").value.trim())}&fmt=${$("tl-fmt").value}`;
      const { ok, j, status } = await post(`/timelapse/start?${q}`);
      if (!ok) { $("tl-stat").textContent = "error: " + (j.detail || status); return; }
      setTl(true);
      if (j.already_running) $("tl-stat").textContent = "already running — " + tlLine(j);
      else { $("tl-dir").value = j.dir; $("tl-stat").textContent = tlLine(j); }
      schedTl();
    } else {
      const { j } = await post("/timelapse/stop");
      setTl(false); clearTimeout(tlTimer);
      $("tl-stat").textContent = "stopped — " + tlLine(j);
    }
  } finally { btn.disabled = false; }
};
async function pollTl() {
  try {
    const j = await (await fetch(BASE + "/timelapse/status")).json();
    if (j.running) {
      if (!tlOn) setTl(true);
      if (!$("tl-dir").value) $("tl-dir").value = j.dir;
      $("tl-stat").textContent = tlLine(j);
    } else if (tlOn) {
      setTl(false);
      $("tl-stat").textContent = (j.taken === undefined)
        ? "server restarted — timelapse state lost (saved files are safe)"
        : "finished — " + tlLine(j);
    }
  } catch (e) { }
  if (tlOn) schedTl();
}

// ------------------------------------------------------------- presets -----
async function refreshPresets() {
  try {
    const l = await (await fetch(BASE + "/presets")).json();
    $("plist").innerHTML = l.map(n => `<option value="${n}">`).join("");
  } catch (e) { }
}
$("psave").onclick = async () => {
  const n = $("pname").value || "default";
  const { ok, j } = await post(`/presets/save?preset=${encodeURIComponent(n)}`);
  stat(ok ? `preset saved → ${j.path}` : `preset: ${j.detail}`);
  refreshPresets();
};
$("pload").onclick = async () => {
  const n = $("pname").value || "default";
  const { ok, j } = await post(`/presets/load?preset=${encodeURIComponent(n)}`);
  stat(ok ? `preset loaded: ${n}` : `preset: ${j.detail || "error"}`);
  if (ok) { cancelDebounces(); loadCaps(); resetStream(); }
};

// ------------------------------------------------------------- conn/info ---
$("conn").onclick = async () => {
  let j = {};
  try { j = await (await fetch(BASE + "/info")).json(); } catch (e) { }
  const ep = j.ok ? "/disconnect" : "/connect";
  stat(j.ok ? "disconnecting (freeing camera)…" : "connecting…");
  const r = await post(ep);
  if (!r.ok) { stat("error: " + (r.j.detail || r.status)); return; }
  setTimeout(() => location.reload(), 500);   // re-establish stream + controls
};

async function pollInfo() {
  try {
    const j = await (await fetch(BASE + "/info")).json();
    if (j.ok) {
      $("info").textContent =
        `${j.model} s/n ${j.serial} · ${j.width}×${j.height} · acq ${j.acquisition_fps.toFixed(1)} fps`;
      $("conn").textContent = "disconnect";
      document.title = `${NAME} · ${j.model}`;
      $("title").textContent = j.model || NAME;
      // Multi-client reconciliation: another tab may have changed the ROI.
      if (j.roi && R.roi && !roisEqual(j.roi, R.roi)) {
        zoomStack.length = 0;
        cancelDebounces();
        await loadCaps();
        resetStream();
      }
    } else {
      $("info").textContent = "disconnected" + (j.error ? " · " + j.error : "");
      $("conn").textContent = "connect";
    }
  } catch (e) { }
  setTimeout(pollInfo, 1000);
}

// ------------------------------------------------------------- boot --------
(async function boot() {
  M = new MeasureLayer({
    media: img, svg, stage: sbox,
    transform: liveTransform,          // sensor pixels — the live/offline bridge
    storageKey: ANNOT_KEY,
    onChange: renderMeasures,
    onToolChange: t => { if (t !== tool) setTool(t); },
  });
  M.renderList($("measure-tbl").querySelector("tbody"));
  M.buildToolbar($("toolbar"), [{
    tool: "roi", glyph: "⛶", label: "ROI",
    title: "drag a box to crop the sensor readout (raises frame rate)",
  }]);
  setTool("roi");
  await loadCaps();
  await loadCal();          // sets umPerPx, then re-renders measurements
  renderMeasures();
  resetStream();
  refreshPresets();
  pollInfo();
  pollRecord();
  pollTl();       // restores a running timelapse after a page reload
})();
