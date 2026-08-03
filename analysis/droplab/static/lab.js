/* droplab — lab page. Frame viewer with drag-ROI (mapping style follows
 * camera_dock/static/cam.js: coordinates are computed against the image's
 * natural size, everything in CSS pixels), parameter panel, background jobs
 * with polling, results gallery. */
"use strict";

// Mount-agnostic: the page lives at <base>/lab/<video>, so <base> is "" when
// droplab is served standalone and "/analysis" when the DAQ panel mounts it.
const _M = location.pathname.replace(/\/+$/, "").match(/^(.*)\/lab\/([^/]+)$/);
const BASE = _M ? _M[1] : "";
const NAME = _M ? decodeURIComponent(_M[2]) : "";
const V = encodeURIComponent(NAME);
const api = p => BASE + p;
const $ = id => document.getElementById(id);
document.title = NAME;
$("title").textContent = NAME;

// ------------------------------------------------------------------ state --
let INFO = null;                 // /video/{name}/info
let cur = 0;                     // current frame index
let viewMode = "raw";            // raw | droplets | processed | sum | sumfull
let roi = null;                  // [x0, x1, y0, y1] image px
let frameSeq = 0;                // async guard for frame loads
let activeJob = null;            // polled job id
let tracksData = null;           // /streamlines payload

// The pixel scale this recording was prefilled with, and where it came from.
// Every physical result downstream — radius, velocity, volume, mass — scales
// with this number, so the caption must describe what is in the field NOW, not
// what was loaded at boot: typing a value, or loading a saved log, must stop it
// claiming to be the recording's own calibration.
const PPUM = { value: null, source: "", trusted: false };

function refreshPpumSrc() {
  const el = $("ppum-src");
  if (!el) return;
  const shown = parseFloat($("p-ppum").value);
  const unchanged = PPUM.value != null && isFinite(shown)
    && Math.abs(shown - PPUM.value) <= Math.abs(PPUM.value) * 1e-9;
  if (unchanged) {
    el.textContent = "scale: " + PPUM.source;
    el.style.color = PPUM.trusted ? "#7f7" : "#fc7";
  } else if (isFinite(shown)) {
    el.textContent = "scale: entered by hand";
    el.style.color = "#fc7";
  } else {
    el.textContent = "scale: not a number";
    el.style.color = "#c77";
  }
}

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.style.display = "block";
  clearTimeout(t._h);
  t._h = setTimeout(() => { t.style.display = "none"; }, 3500);
}
function stat(msg) { $("stat").textContent = msg; }

const num = (id, fallback) => {
  const v = parseFloat($(id).value);
  return isFinite(v) ? v : fallback;
};

// ------------------------------------------------------------- parameters --
function detectParams() {
  return {
    threshold: Math.round(num("p-threshold", 50)),
    min_area: Math.round(num("p-min-area", 10)),
    max_area: Math.round(num("p-max-area", 100)),
    start_time: num("p-start", 0),
    end_time: num("p-end", 0),
    fps_override: $("p-fps").value.trim() || null,
    exposure_time_s: $("p-exp").value.trim() || null,
    // Fall back to this recording's own calibration, not the foreign constant.
    pixels_per_um: num("p-ppum", PPUM.value || 0.1878),
    x_start: roi[0], x_stop: roi[1], y_start: roi[2], y_stop: roi[3],
  };
}

function roiQuery() {
  return `x0=${roi[0]}&x1=${roi[1]}&y0=${roi[2]}&y1=${roi[3]}`;
}

function drawRoiFields() {
  $("roi-x0").value = roi[0]; $("roi-x1").value = roi[1];
  $("roi-y0").value = roi[2]; $("roi-y1").value = roi[3];
}

function setRoi(x0, x1, y0, y1) {
  const W = INFO.width, H = INFO.height;
  x0 = Math.max(0, Math.min(Math.round(x0), W - 1));
  x1 = Math.max(x0 + 1, Math.min(Math.round(x1), W));
  y0 = Math.max(0, Math.min(Math.round(y0), H - 1));
  y1 = Math.max(y0 + 1, Math.min(Math.round(y1), H));
  roi = [x0, x1, y0, y1];
  drawRoiFields();
  showFrame();
}

// ------------------------------------------------------------ frame views --
const img = $("view");
let lastUrl = null;

function clearComputingStat() {
  if ($("stat").textContent.startsWith("computing persistence")) stat("");
}

async function showFrame() {
  if (!INFO) return;
  const seq = ++frameSeq;
  const p = detectParams();
  let url;
  clearComputingStat();                              // stale message from a superseded sum
  if (viewMode === "sum" || viewMode === "sumfull") {
    const fps = parseFloat(p.fps_override) || INFO.fps || 30;
    const s = Math.floor(p.start_time * fps), e = Math.floor(p.end_time * fps);
    url = api(`/video/${V}/sum?start=${s}&end=${e}` +
              `&threshold=${p.threshold}&mode=${viewMode === "sum" ? "analysis" : "full"}&${roiQuery()}`);
    stat("computing persistence view…");
  } else {
    url = api(`/video/${V}/frame/${cur}?view=${viewMode}` +
              `&threshold=${p.threshold}&min_area=${p.min_area}&max_area=${p.max_area}&${roiQuery()}`);
  }
  try {
    const r = await fetch(url);
    if (seq !== frameSeq) return;                    // superseded
    if (!r.ok) {
      clearComputingStat();
      let msg = r.status;
      try { msg = (await r.json()).detail || msg; } catch (e) { }
      toast(`frame: ${msg}`);
      return;
    }
    const blob = await r.blob();
    if (seq !== frameSeq) return;
    const u = URL.createObjectURL(blob);
    img.onload = () => { if (lastUrl) URL.revokeObjectURL(lastUrl); lastUrl = u; drawTracks(); };
    img.src = u;
    const n = r.headers.get("X-Droplet-Count");
    const nr = r.headers.get("X-Droplet-Count-Roi");
    $("count").textContent = n !== null ? `droplets: ${n} (${nr} in ROI)` : "";
    if (viewMode === "sum" || viewMode === "sumfull") {
      const step = r.headers.get("X-Sum-Step");
      stat(step > 1 ? `persistence view (every ${step}th frame)` : "");
    }
  } catch (e) {
    if (seq === frameSeq) { clearComputingStat(); toast("frame load failed"); }
  }
}

let frameDebounce = null;
function refreshSoon() {
  clearTimeout(frameDebounce);
  frameDebounce = setTimeout(showFrame, 150);
}

function goto(idx) {
  cur = Math.max(0, Math.min(idx, (INFO ? INFO.total_frames : 1) - 1));
  $("frame-num").value = cur;
  $("scrub").value = cur;
  showFrame();
}

// view mode buttons
const MODES = { "vm-raw": "raw", "vm-droplets": "droplets", "vm-processed": "processed",
                "vm-sum": "sum", "vm-sumfull": "sumfull" };
for (const [id, mode] of Object.entries(MODES)) {
  $(id).onclick = () => {
    viewMode = mode;
    for (const other of Object.keys(MODES)) $(other).classList.toggle("active", other === id);
    showFrame();
  };
}

// transport
$("b-back100").onclick = () => goto(cur - 100);
$("b-back10").onclick = () => goto(cur - 10);
$("b-back1").onclick = () => goto(cur - 1);
$("b-fwd1").onclick = () => goto(cur + 1);
$("b-fwd10").onclick = () => goto(cur + 10);
$("b-fwd100").onclick = () => goto(cur + 100);
$("scrub").oninput = () => { cur = parseInt($("scrub").value, 10); $("frame-num").value = cur; refreshSoon(); };
$("frame-num").onchange = () => goto(parseInt($("frame-num").value, 10) || 0);
document.addEventListener("keydown", ev => {
  if (ev.target.tagName === "INPUT" || ev.target.tagName === "SELECT") return;
  if (ev.key === "ArrowLeft") goto(cur - 1);
  else if (ev.key === "ArrowRight") goto(cur + 1);
  else if (ev.key === "PageUp") goto(cur + 100);
  else if (ev.key === "PageDown") goto(cur - 100);
  else return;
  ev.preventDefault();                   // keep the page from scrolling away
});

// parameter edits refresh the preview
for (const id of ["p-threshold", "p-min-area", "p-max-area", "p-start", "p-end"])
  $(id).addEventListener("input", refreshSoon);

// p-ppum changes nothing on screen, but it must never keep claiming to be the
// recording's calibration once someone has typed over it.
$("p-ppum").addEventListener("input", refreshPpumSrc);

// ------------------------------------------------------------- drag-to-ROI --
const box = $("dragbox"), vbox = $("viewbox");
img.addEventListener("dragstart", ev => ev.preventDefault());
let drag = null;

function contentRect() {
  const r = img.getBoundingClientRect();
  const bl = img.clientLeft, bt = img.clientTop;
  return { left: r.left + bl, top: r.top + bt, w: r.width - 2 * bl, h: r.height - 2 * bt };
}

vbox.addEventListener("pointerdown", ev => {
  if (ev.button !== 0 || !INFO) return;
  // Staleness gate (as cam.js): only map coordinates against a real frame,
  // never the placeholder box shown before the first image loads.
  if (img.naturalWidth !== INFO.width || img.naturalHeight !== INFO.height) {
    toast("frame still loading — try again");
    return;
  }
  drag = { x0: ev.clientX, y0: ev.clientY, rect: contentRect() };
  try { vbox.setPointerCapture(ev.pointerId); } catch (e) { }
  ev.preventDefault();
});
vbox.addEventListener("pointermove", ev => {
  if (!drag) return;
  const { rect } = drag;
  const x = Math.min(Math.max(ev.clientX, rect.left), rect.left + rect.w);
  const y = Math.min(Math.max(ev.clientY, rect.top), rect.top + rect.h);
  const bx = vbox.getBoundingClientRect();
  box.style.display = "block";
  box.style.left = (Math.min(drag.x0, x) - bx.left) + "px";
  box.style.top = (Math.min(drag.y0, y) - bx.top) + "px";
  box.style.width = Math.abs(x - drag.x0) + "px";
  box.style.height = Math.abs(y - drag.y0) + "px";
});
vbox.addEventListener("pointerup", ev => {
  if (!drag) return;
  box.style.display = "none";
  const d = drag; drag = null;
  const { rect } = d;
  const x1 = Math.min(Math.max(ev.clientX, rect.left), rect.left + rect.w);
  const y1 = Math.min(Math.max(ev.clientY, rect.top), rect.top + rect.h);
  if (Math.abs(x1 - d.x0) < 8 || Math.abs(y1 - d.y0) < 8) return;   // accidental click
  const W = INFO.width, H = INFO.height;
  const u0 = Math.min(d.x0, x1) - rect.left, u1 = Math.max(d.x0, x1) - rect.left;
  const v0 = Math.min(d.y0, y1) - rect.top, v1 = Math.max(d.y0, y1) - rect.top;
  setRoi(u0 * W / rect.w, u1 * W / rect.w, v0 * H / rect.h, v1 * H / rect.h);
});
vbox.addEventListener("contextmenu", ev => {
  ev.preventDefault();
  if (INFO) setRoi(0, INFO.width, 0, INFO.height);
});
$("roi-apply").onclick = () => setRoi(
  parseInt($("roi-x0").value, 10) || 0, parseInt($("roi-x1").value, 10) || INFO.width,
  parseInt($("roi-y0").value, 10) || 0, parseInt($("roi-y1").value, 10) || INFO.height);
$("roi-full").onclick = () => setRoi(0, INFO.width, 0, INFO.height);

// -------------------------------------------------------------- tracks overlay
const tcanvas = $("tracks-canvas");
async function refreshTracks() {
  try {
    const r = await fetch(api(`/video/${V}/streamlines`));
    if (!r.ok) { toast("no track data — run tracking first"); $("show-tracks").checked = false; return; }
    tracksData = await r.json();
  } catch (e) { return; }
  if (tracksData.truncated) {
    stat(`tracks overlay: showing the ${tracksData.tracks.length} longest of ` +
         `${tracksData.n_tracks_total} tracks`);
  }
  drawTracks();
}
$("show-tracks").onchange = () => {
  if ($("show-tracks").checked && tracksData === null) { refreshTracks(); return; }
  drawTracks();
};

function drawTracks() {
  const show = $("show-tracks").checked && tracksData && INFO;
  // A hidden pane (the DAQ panel hides inactive tabs with display:none)
  // reports a zero-sized image; resizing the canvas to 0 would wipe the
  // overlay for good, since no resize event fires when the pane comes back.
  if (!img.clientWidth || !img.clientHeight) return;
  tcanvas.width = img.clientWidth;
  tcanvas.height = img.clientHeight;
  const ctx = tcanvas.getContext("2d");
  ctx.clearRect(0, 0, tcanvas.width, tcanvas.height);
  if (!show) return;
  const sx = img.clientWidth / INFO.width, sy = img.clientHeight / INFO.height;
  const palette = ["#f55", "#5cf", "#5f5", "#fd5", "#d7f", "#f95", "#5ff", "#f5b"];
  tracksData.tracks.forEach((t, i) => {
    if (t.points.length < 2) return;
    ctx.strokeStyle = palette[i % palette.length];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(t.points[0][0] * sx, t.points[0][1] * sy);
    for (let k = 1; k < t.points.length; k++)
      ctx.lineTo(t.points[k][0] * sx, t.points[k][1] * sy);
    ctx.stroke();
  });
}
window.addEventListener("resize", drawTracks);
// window resize never fires for a pane being shown/hidden inside the DAQ
// panel — observing the element itself does (same trick as the dock's
// measure.js overlay canvas).
if (window.ResizeObserver) new ResizeObserver(() => drawTracks()).observe(img);

// ------------------------------------------------------------------- jobs --
async function postJob(url, body, label) {
  try {
    const r = await fetch(api(url), { method: "POST", headers: { "Content-Type": "application/json" },
                                      body: JSON.stringify(body) });
    const j = await r.json();
    if (!r.ok) { toast(`${label}: ${j.detail || r.status}`); return; }
    activeJob = j.id;
    $("job-prog").style.display = "";
    $("job-cancel").style.display = "";
    stat(`${label} started…`);
    pollJob();
  } catch (e) { toast(`${label} failed: ${e}`); }
}

async function pollJob() {
  if (!activeJob) return;
  let j;
  try {
    const r = await fetch(api(`/jobs/${activeJob}`));
    if (!r.ok) { setTimeout(pollJob, 1500); return; }
    j = await r.json();
  }
  catch (e) { setTimeout(pollJob, 1500); return; }
  $("job-prog").value = j.progress || 0;
  stat(`${j.kind}: ${j.state}` + (j.message ? ` — ${j.message}` : "") +
       (j.state === "running" ? ` (${Math.round((j.progress || 0) * 100)}%)` : ""));
  if (j.state === "queued" || j.state === "running") {
    setTimeout(pollJob, 700);
    return;
  }
  activeJob = null;
  $("job-prog").style.display = "none";
  $("job-cancel").style.display = "none";
  if (j.state === "done") {
    stat(`${j.kind} done in ${j.elapsed_s}s`);
    if (j.kind === "track") {
      tracksData = null;
      if ($("show-tracks").checked) refreshTracks();  // keep overlay live
    }
    loadResults();
    loadLogs();
  } else if (j.state === "error") {
    stat(`${j.kind} FAILED: ${j.error}`);
    toast(j.error || "job failed");
  } else {
    stat(`${j.kind} ${j.state}`);
  }
}

$("job-cancel").onclick = () => { if (activeJob) fetch(api(`/jobs/${activeJob}/cancel`), { method: "POST" }); };

$("run-detect").onclick = () => {
  const p = detectParams();
  if (p.end_time <= p.start_time) { toast("end time must be after start time"); return; }
  postJob(`/video/${encodeURIComponent(NAME)}/analyze`, p, "detection");
};
$("run-track").onclick = () => postJob(`/video/${encodeURIComponent(NAME)}/track`, {
  max_distance: num("t-dist", 24), auto_distance: $("t-auto").checked,
  min_track_length: Math.round(num("t-minlen", 1)),
}, "tracking");
$("run-trackan").onclick = () => postJob(`/video/${encodeURIComponent(NAME)}/track_analysis`, {}, "track analysis");
$("run-dist").onclick = () => postJob(`/video/${encodeURIComponent(NAME)}/distributions`, {
  density_kg_m3: num("d-density", 3520),
}, "distributions");
$("run-long").onclick = () => postJob(`/video/${encodeURIComponent(NAME)}/longitudinal`, {
  chunk_duration_s: num("l-chunk", 5),
  // The server reads this too (server.py:473). Omitting it meant Longitudinal
  // silently used 3520 while Distributions used whatever was typed above —
  // two stages of one analysis disagreeing about the material.
  density_kg_m3: num("d-density", 3520),
}, "longitudinal");

// -------------------------------------------------------------- load a log --
async function loadLogs() {
  try {
    const logs = await (await fetch(api(`/video/${V}/logs`))).json();
    $("log-list").innerHTML = '<option value="">— saved logs —</option>' +
      logs.map(l => `<option value="${l.name}">${l.name}</option>`).join("");
  } catch (e) { }
}
$("log-load").onclick = async () => {
  const name = $("log-list").value;
  if (!name) return;
  try {
    const log = await (await fetch(api(`/log?path=${encodeURIComponent(name)}`))).json();
    const p = log.analysis_parameters || {};
    if (p.threshold != null) $("p-threshold").value = p.threshold;
    if (p.min_area != null) $("p-min-area").value = p.min_area;
    if (p.max_area != null) $("p-max-area").value = p.max_area;
    if (p.start_time != null) $("p-start").value = p.start_time;
    if (p.end_time != null) $("p-end").value = p.end_time;
    $("p-fps").value = p.fps_override != null ? p.fps_override : "";
    $("p-exp").value = p.exposure_time_s != null ? p.exposure_time_s : "";
    if (p.pixels_per_um != null) $("p-ppum").value = p.pixels_per_um;
    refreshPpumSrc();      // a log's scale is the log's, not this recording's
    if (p.x_start != null) setRoi(p.x_start, p.x_stop, p.y_start, p.y_stop);
    else showFrame();
    stat(`loaded ${name}`);
  } catch (e) { toast("could not load log"); }
};

// --------------------------------------------------------------- results --
function statline(label, value) {
  return `<div class="statline">${label}: ${value}</div>`;
}

async function loadResults() {
  let res;
  try { res = await (await fetch(api(`/video/${V}/results`))).json(); }
  catch (e) { return; }
  const parts = [];
  if (res.detection) {
    const r = res.detection.results || {};
    const ap = res.detection.analysis_parameters || {};
    parts.push(statline("frames analyzed", r.frames_analyzed));
    parts.push(statline("mean count (ROI)", `${(r.mean_droplet_count_roi ?? 0).toFixed(2)} ± ${(r.std_droplet_count_roi ?? 0).toFixed(2)}`));
    parts.push(statline("count range (ROI)", `${r.min_droplet_count_roi} – ${r.max_droplet_count_roi}`));
    parts.push(statline("density (ROI beam)", `${(r.mean_density_roi_per_pixel ?? 0).toExponential(3)} /px²`));
    parts.push(statline("window", `${ap.start_time}s – ${ap.end_time}s @ ${(ap.fps ?? 0).toFixed(2)} fps`));
    parts.push(statline("ROI", `x ${ap.x_start}–${ap.x_stop}, y ${ap.y_start}–${ap.y_stop}`));
  }
  if (res.tracking) {
    parts.push(statline("tracks", `${res.tracking.n_tracks} (max dist ${res.tracking.max_distance} px, ${res.tracking.n_observations} obs)`));
  }
  if (res.longitudinal_files && res.longitudinal_files.length) {
    parts.push(statline("longitudinal runs", res.longitudinal_files.length));
  }
  $("res-stats").innerHTML = parts.join("") || '<span class="small">nothing analyzed yet</span>';

  const gal = (res.plots || []).map(f =>
    `<a href="${api(`/plots/${encodeURIComponent(res.basename)}/${encodeURIComponent(f)}`)}" target="_blank">
       <img src="${api(`/plots/${encodeURIComponent(res.basename)}/${encodeURIComponent(f)}`)}?t=${Date.now()}">
       <div class="cap">${f}</div></a>`);
  $("gallery").innerHTML = gal.join("") || '<span class="small">—</span>';
}

// ------------------------------------------------------------------ boot --
(async function boot() {
  let info;
  try {
    const r = await fetch(api(`/video/${V}/info`));
    if (!r.ok) {
      let msg = r.status;
      try { msg = (await r.json()).detail || msg; } catch (e) { }
      toast(`video info: ${msg}`);
      return;
    }
    info = await r.json();
  }
  catch (e) { toast("could not load video info"); return; }
  INFO = info;
  $("info").textContent =
    `${info.width}×${info.height} · ${info.total_frames} frames · ` +
    `container ${info.container_fps ? info.container_fps.toFixed(2) : "?"} fps` +
    (info.sidecar_fps ? ` · sidecar ${info.sidecar_fps.toFixed(2)} fps` : "");
  const d = info.defaults;
  $("p-threshold").value = d.threshold;
  $("p-min-area").value = d.min_area;
  $("p-max-area").value = d.max_area;
  $("p-start").value = d.start_time;
  $("p-end").value = d.end_time;
  if (d.fps_override != null) $("p-fps").value = d.fps_override;
  PPUM.value = d.pixels_per_um;
  PPUM.source = d.pixels_per_um_source || "";
  PPUM.trusted = !!d.pixels_per_um_from_sidecar;
  $("p-ppum").value = d.pixels_per_um;
  refreshPpumSrc();
  roi = [0, info.width, 0, info.height];
  drawRoiFields();
  $("scrub").max = Math.max(info.total_frames - 1, 0);
  goto(0);
  loadResults();
  loadLogs();
})();
