/* measure.js — shared annotation + measurement layer.
 * =====================================================================
 * SINGLE SOURCE OF TRUTH for measuring, used by BOTH:
 *   · live analysis  — camera_dock/static/cam.js (MJPEG stream from a camera)
 *   · offline analysis — camera_dock/static/measure.html (a still or a video
 *     file), and whatever the analysis app builds on top of it.
 *
 * Zero dependencies, no build step, no imports: drop in with a <script> tag and
 * use `window.MeasureLayer`. Do not fork this file — see docs/MEASURE_INTEGRATION.md.
 *
 * ---------------------------------------------------------------------
 * THE ONE IDEA: annotations are stored in NATIVE pixel coordinates.
 *
 * "Native" means the pixel grid of the thing being measured:
 *   · offline — pixels of the image/video file
 *   · live    — pixels of the camera SENSOR (not of the streamed image)
 *
 * Everything the user sees (positions on screen) is derived from native coords
 * through `transform`, never stored. That is why a measurement does not change
 * when you resize the window, and — in the live app — why it stays pinned to
 * the scene when the sensor ROI is cropped or binning changes.
 *
 * The host supplies the transform. Offline needs nothing (the default maps the
 * media's natural pixels onto its displayed box). Live passes a transform that
 * also accounts for the ROI origin and binning. THAT IS THE WHOLE BRIDGE
 * between online and offline — one small object, see `defaultTransform`.
 *
 * ---------------------------------------------------------------------
 * NOTES BETWEEN DEVELOPERS
 * This file is edited by both the live-dock side and the offline-analysis side.
 * When you change something the other side must know about, leave a comment:
 *
 *     // NOTE[offline->live]: ...   (analysis dev -> camera dock dev)
 *     // NOTE[live->offline]: ...   (camera dock dev -> analysis dev)
 *
 * Keep changes generic: anything camera-specific belongs in cam.js, anything
 * file/video-specific belongs in the offline host page. If you need a new hook,
 * add an option with a sane default so the other host keeps working untouched.
 */
"use strict";

(function (global) {
  const SVGNS = "http://www.w3.org/2000/svg";

  // Tools this layer owns. A host may define its own tools (the live dock adds
  // "roi"); when such a tool is active the layer stays out of the way.
  const DRAW_TOOLS = ["line", "circle", "rect"];

  const TOOL_META = {
    line:   { glyph: "╱", label: "line",   hint: "drag: measure a line" },
    circle: { glyph: "◯", label: "circle", hint: "drag from the centre outward" },
    rect:   { glyph: "▭", label: "rect",   hint: "drag: measure a rectangle" },
  };

  function num(v) {
    const a = Math.abs(v);
    return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toFixed(2);
  }

  function el(tag, attrs, cls) {
    const e = document.createElementNS(SVGNS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (cls) e.setAttribute("class", cls);
    return e;
  }

  /* Natural (intrinsic) size of an <img>, <video> or <canvas>. */
  function naturalSize(media) {
    return {
      w: media.naturalWidth || media.videoWidth || media.width || 0,
      h: media.naturalHeight || media.videoHeight || media.height || 0,
    };
  }

  /* The displayed content box of the media, EXCLUDING any CSS border, in CSS px.
   * Everything on-screen is measured against this. */
  function contentRect(media) {
    const r = media.getBoundingClientRect();
    const bl = media.clientLeft || 0, bt = media.clientTop || 0;
    return { left: r.left + bl, top: r.top + bt,
             w: r.width - 2 * bl, h: r.height - 2 * bt };
  }

  /* Default transform: native pixels are the media's own pixels, stretched to
   * the displayed box. This is all the offline host needs.
   *
   * The live host replaces this with one that also shifts by the sensor ROI
   * origin and scales by binning — see cam.js `liveTransform`. */
  function defaultTransform(media) {
    return {
      ready() { const n = naturalSize(media); return n.w > 0 && n.h > 0; },
      imageSize() { const n = naturalSize(media); return [n.w, n.h]; },
      toDisplay(x, y) {
        const rect = contentRect(media), n = naturalSize(media);
        return { u: x * rect.w / n.w, v: y * rect.h / n.h };
      },
      toNative(u, v) {
        const rect = contentRect(media), n = naturalSize(media);
        return { x: u * n.w / rect.w, y: v * n.h / rect.h };
      },
    };
  }

  class MeasureLayer {
    /**
     * @param {object} o
     * @param {HTMLElement} o.media   <img>/<video>/<canvas> being measured
     * @param {SVGElement}  o.svg     absolutely-positioned overlay over `media`
     * @param {HTMLElement} [o.stage] element receiving pointer events (default:
     *                                media.parentNode)
     * @param {object}   [o.transform]  native<->display mapping (see above)
     * @param {string}   [o.storageKey] localStorage key for auto-persistence
     * @param {function} [o.onChange]   called with the annotation array
     * @param {function} [o.onToolChange] called with the active tool name
     */
    constructor(o) {
      this.media = o.media;
      this.svg = o.svg;
      this.stage = o.stage || o.media.parentNode;
      this.transform = o.transform || defaultTransform(o.media);
      this.storageKey = o.storageKey || null;
      this.onChange = o.onChange || function () {};
      this.onToolChange = o.onToolChange || function () {};

      this.annots = [];
      this.seq = 1;
      this.tool = DRAW_TOOLS[0];
      this.umPerPx = null;
      this.scaleLabel = "";
      this.highlightId = null;
      this._draw = null;
      this._laidOut = null;
      this._listEl = null;

      this.load();
      this._bindPointer();

      if (global.ResizeObserver)
        new ResizeObserver(() => this.redraw()).observe(this.media);
      // A live stream can change its intrinsic aspect asynchronously (new ROI)
      // and ResizeObserver is not reliable for that in Chromium, so also poll
      // for drift. Four property reads, five times a second.
      this._timer = setInterval(() => this._checkDrift(), 200);
    }

    destroy() { clearInterval(this._timer); }

    /* ---------------------------------------------------------------- tools */
    owns(tool) { return DRAW_TOOLS.indexOf(tool) >= 0; }

    setTool(t) {
      this.tool = t;
      if (this._toolbar)
        for (const b of this._toolbar.querySelectorAll("[data-tool]"))
          b.classList.toggle("active", b.dataset.tool === t);
      this.onToolChange(t);
    }

    hint() {
      const m = TOOL_META[this.tool];
      return m ? m.hint : "";
    }

    /* Build the toolbar into `host`. `extra` lets the host prepend its own
     * tools, e.g. the live dock's ROI tool:
     *   [{tool: 'roi', glyph: '⛶', label: 'ROI', title: '...'}]
     * Host tools are rendered and highlighted here but their behaviour is the
     * host's business (this layer ignores pointer events while one is active).
     */
    buildToolbar(host, extra) {
      this._toolbar = host;
      host.innerHTML = "";
      const mk = (tool, glyph, label, title) => {
        const b = document.createElement("button");
        b.className = "tool";
        b.dataset.tool = tool;
        b.textContent = `${glyph} ${label}`;
        if (title) b.title = title;
        b.onclick = () => this.setTool(tool);
        host.appendChild(b);
        return b;
      };
      for (const e of (extra || [])) mk(e.tool, e.glyph, e.label, e.title);
      if ((extra || []).length) host.appendChild(this._sep());
      for (const t of DRAW_TOOLS) mk(t, TOOL_META[t].glyph, TOOL_META[t].label,
                                     TOOL_META[t].hint);
      host.appendChild(this._sep());
      const undo = document.createElement("button");
      undo.textContent = "undo"; undo.title = "remove the last annotation";
      undo.onclick = () => this.undo();
      host.appendChild(undo);
      const clear = document.createElement("button");
      clear.textContent = "clear"; clear.title = "remove every annotation";
      clear.onclick = () => this.clear();
      host.appendChild(clear);
      host.appendChild(this._sep());
      this._badge = document.createElement("span");
      this._badge.className = "badge";
      host.appendChild(this._badge);
      this._renderBadge();
      this.setTool(this.tool);
      return host;
    }

    _sep() {
      const s = document.createElement("span");
      s.className = "sep";
      return s;
    }

    /* ---------------------------------------------------------- calibration */
    /* `umPerPx` is micrometres per NATIVE pixel (sensor pixel when live), so it
     * survives binning and ROI changes. Pass null to show pixels only. */
    setScale(umPerPx, label) {
      this.umPerPx = (umPerPx && isFinite(umPerPx) && umPerPx > 0)
        ? Number(umPerPx) : null;
      this.scaleLabel = label || "";
      this._renderBadge();
      this.redraw();
      this.renderList();
    }

    _renderBadge() {
      if (!this._badge) return;
      this._badge.textContent = this.umPerPx
        ? `${this.scaleLabel ? this.scaleLabel + ": " : ""}${num(this.umPerPx)} µm/px`
        : "no calibration";
      this._badge.classList.toggle("on", !!this.umPerPx);
    }

    fmtLen(px) {
      const p = `${num(px)} px`;
      return this.umPerPx ? `${p} · ${num(px * this.umPerPx)} µm` : p;
    }
    fmtPair(w, h) {
      const p = `${num(w)} × ${num(h)} px`;
      return this.umPerPx
        ? `${p} · ${num(w * this.umPerPx)} × ${num(h * this.umPerPx)} µm` : p;
    }

    /* Measurement text. All maths is in native px, so numbers never change with
     * zoom, window size, or (live) an ROI crop. */
    describe(a) {
      const dx = a.x1 - a.x0, dy = a.y1 - a.y0;
      if (a.type === "line") return { short: this.fmtLen(Math.hypot(dx, dy)) };
      if (a.type === "rect")
        return { short: this.fmtPair(Math.abs(dx), Math.abs(dy)) };
      const r = Math.hypot(dx, dy);
      return { short: `⌀ ${this.fmtLen(2 * r)}`, extra: `r ${this.fmtLen(r)}` };
    }

    static lineLength(a) { return Math.hypot(a.x1 - a.x0, a.y1 - a.y0); }

    /* ------------------------------------------------------------ rendering */
    _layout() {
      const rect = contentRect(this.media);
      this.svg.style.left = (this.media.clientLeft || 0) + "px";
      this.svg.style.top = (this.media.clientTop || 0) + "px";
      this.svg.setAttribute("width", rect.w);
      this.svg.setAttribute("height", rect.h);
      this.svg.setAttribute("viewBox", `0 0 ${rect.w} ${rect.h}`);
      this._laidOut = { w: rect.w, h: rect.h };
      return rect;
    }

    _checkDrift() {
      if (!this._laidOut || !this.transform.ready()) return;
      const rect = contentRect(this.media);
      if (Math.abs(rect.w - this._laidOut.w) > 0.5 ||
          Math.abs(rect.h - this._laidOut.h) > 0.5) this.redraw();
    }

    _shape(a, cls) {
      const p0 = this.transform.toDisplay(a.x0, a.y0);
      const p1 = this.transform.toDisplay(a.x1, a.y1);
      const g = document.createDocumentFragment();
      const hl = (a.id && a.id === this.highlightId) ? " hl" : "";
      const sc = `shape${hl}${cls ? " " + cls : ""}`;
      let lx, ly;
      if (a.type === "line") {
        g.appendChild(el("line", { x1: p0.u, y1: p0.v, x2: p1.u, y2: p1.v }, sc));
        g.appendChild(el("circle", { cx: p0.u, cy: p0.v, r: 2.5 }, "handle"));
        g.appendChild(el("circle", { cx: p1.u, cy: p1.v, r: 2.5 }, "handle"));
        lx = (p0.u + p1.u) / 2; ly = (p0.v + p1.v) / 2 - 7;
      } else if (a.type === "rect") {
        const x = Math.min(p0.u, p1.u), y = Math.min(p0.v, p1.v);
        const w = Math.abs(p1.u - p0.u), h = Math.abs(p1.v - p0.v);
        g.appendChild(el("rect", { x, y, width: w, height: h }, sc));
        lx = x + w / 2; ly = y - 5;
      } else {
        const r = Math.hypot(p1.u - p0.u, p1.v - p0.v);
        g.appendChild(el("circle", { cx: p0.u, cy: p0.v, r }, sc));
        g.appendChild(el("circle", { cx: p0.u, cy: p0.v, r: 2 }, "handle"));
        lx = p0.u; ly = p0.v - r - 5;
      }
      const rect = contentRect(this.media);
      const t = el("text", { x: Math.min(Math.max(lx, 34), rect.w - 34),
                             y: Math.min(Math.max(ly, 12), rect.h - 4),
                             "text-anchor": "middle" }, hl ? "hl" : null);
      t.textContent = this.describe(a).short;
      g.appendChild(t);
      return g;
    }

    redraw() {
      if (!this.svg) return;
      while (this.svg.firstChild) this.svg.removeChild(this.svg.firstChild);
      if (!this.transform.ready()) return;
      this._layout();
      for (const a of this.annots) this.svg.appendChild(this._shape(a));
      const d = this._draw;
      if (d && d.cur) this.svg.appendChild(this._shape(
        { type: d.type, x0: d.x0, y0: d.y0, x1: d.cur.x, y1: d.cur.y }, "preview"));
    }

    /* Render the measurement table into a <tbody>. Remembered, so later edits
     * refresh it automatically. */
    renderList(tbody) {
      if (tbody) this._listEl = tbody;
      const body = this._listEl;
      if (!body) return;
      body.innerHTML = "";
      for (const a of this.annots) {
        const d = this.describe(a);
        const tr = document.createElement("tr");
        tr.innerHTML =
          `<td class="mt">${(TOOL_META[a.type] || {}).glyph || "?"}</td>` +
          `<td>${d.short}${d.extra ? `<div class="mu">${d.extra}</div>` : ""}</td>` +
          `<td class="mx" title="delete">✕</td>`;
        tr.onmouseenter = () => { this.highlightId = a.id; this.redraw(); };
        tr.onmouseleave = () => { this.highlightId = null; this.redraw(); };
        tr.querySelector(".mx").onclick = () => this.remove(a.id);
        body.appendChild(tr);
      }
    }

    /* ------------------------------------------------------------- editing */
    add(a) {
      a.id = this.seq++;
      this.annots.push(a);
      this._changed();
      return a;
    }
    remove(id) {
      this.annots = this.annots.filter(x => x.id !== id);
      this._changed();
    }
    undo() { this.annots.pop(); this._changed(); }
    clear() { this.annots = []; this._changed(); }

    get lines() { return this.annots.filter(a => a.type === "line"); }
    byId(id) { return this.annots.find(a => a.id === id); }

    _changed() {
      this.save();
      this.redraw();
      this.renderList();
      this.onChange(this.annots);
    }

    /* --------------------------------------------------------- persistence */
    save() {
      if (!this.storageKey) return;
      try { localStorage.setItem(this.storageKey, JSON.stringify(this.annots)); }
      catch (e) { /* private mode / quota — measurements are not precious */ }
    }
    load() {
      if (!this.storageKey) return;
      try {
        const raw = localStorage.getItem(this.storageKey);
        if (raw) this.import(JSON.parse(raw));
      } catch (e) { this.annots = []; }
    }
    /* Portable JSON — hand this to analysis code, or re-import it later. */
    export() {
      return { schema: "xsphere.measure.v1", um_per_px: this.umPerPx,
               annotations: this.annots };
    }
    import(data) {
      const list = Array.isArray(data) ? data : (data && data.annotations) || [];
      this.annots = list.filter(a => a && TOOL_META[a.type]);
      this.seq = this.annots.reduce((m, a) => Math.max(m, a.id || 0), 0) + 1;
      if (data && data.um_per_px) this.setScale(data.um_per_px, "imported");
      this.redraw();
      this.renderList();
    }

    /* ----------------------------------------------------------- pointers */
    _bindPointer() {
      const stage = this.stage;
      if (this.media.tagName === "IMG") this.media.draggable = false;
      this.media.addEventListener("dragstart", ev => ev.preventDefault());

      stage.addEventListener("pointerdown", ev => {
        if (ev.button !== 0 || !this.owns(this.tool)) return;   // host's tool
        if (!this.transform.ready()) return;
        const rect = contentRect(this.media);
        const s = this.transform.toNative(ev.clientX - rect.left,
                                          ev.clientY - rect.top);
        this._draw = { type: this.tool, x0: s.x, y0: s.y, cur: null,
                       epoch: this.epoch };
        try { stage.setPointerCapture(ev.pointerId); } catch (e) {}
        ev.preventDefault();
      });

      stage.addEventListener("pointermove", ev => {
        if (!this._draw) return;
        const rect = contentRect(this.media);
        this._draw.cur = this.transform.toNative(ev.clientX - rect.left,
                                                 ev.clientY - rect.top);
        this.redraw();
      });

      const finish = () => {
        const d = this._draw;
        this._draw = null;
        if (!d || !d.cur) { this.redraw(); return; }
        // Discard a stray click, and (live) a drag whose geometry moved under it.
        const moved = this.epoch === d.epoch;
        const p0 = this.transform.toDisplay(d.x0, d.y0);
        const p1 = this.transform.toDisplay(d.cur.x, d.cur.y);
        if (moved && Math.hypot(p1.u - p0.u, p1.v - p0.v) >= 6)
          this.add({ type: d.type, x0: d.x0, y0: d.y0, x1: d.cur.x, y1: d.cur.y });
        else this.redraw();
      };
      stage.addEventListener("pointerup", finish);
      stage.addEventListener("pointercancel", finish);
    }
  }

  /* The host bumps `epoch` whenever the native<->display mapping changes under
   * the user (the live dock does this on every ROI change) so an in-flight drag
   * is discarded instead of landing in the wrong place. Offline never changes. */
  MeasureLayer.prototype.epoch = 0;

  MeasureLayer.DRAW_TOOLS = DRAW_TOOLS;
  MeasureLayer.TOOL_META = TOOL_META;
  MeasureLayer.contentRect = contentRect;
  MeasureLayer.naturalSize = naturalSize;
  MeasureLayer.defaultTransform = defaultTransform;
  MeasureLayer.num = num;

  global.MeasureLayer = MeasureLayer;
})(window);
