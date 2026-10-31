// Small dependency-free SVG chart kit: time-series line/band chart, horizontal
// bar chart and a month-by-year calendar heatmap. Every chart has a hover +
// keyboard crosshair tooltip; tables provide the non-visual view.

const NS = "http://www.w3.org/2000/svg";
let uid = 0;

export type Month = string; // "YYYY-MM"

export const monthToX = (m: Month): number => {
  const [y, mo] = m.split("-").map(Number);
  return y + (mo - 1) / 12;
};
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
export const fmtMonth = (m: Month): string => {
  const [y, mo] = m.split("-").map(Number);
  return `${MONTHS[mo - 1]} ${y}`;
};
export const fmt = (v: number | null | undefined, nd = 2): string =>
  v == null || !Number.isFinite(v) ? "—" : (v < 0 ? "−" : "") + Math.abs(v).toFixed(nd);
export const fmtSigned = (v: number, nd = 2): string => (v > 0 ? "+" : "") + fmt(v, nd);

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}, parent?: Element) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  parent?.appendChild(n);
  return n;
}

const tooltip = () => document.getElementById("tooltip") as HTMLDivElement;
export function showTip(html: string, x: number, y: number) {
  const t = tooltip();
  t.innerHTML = html;
  t.hidden = false;
  const r = t.getBoundingClientRect();
  let left = x + 14;
  if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
  let top = y - r.height - 12;
  if (top < 8) top = y + 16;
  t.style.left = `${Math.max(8, left)}px`;
  t.style.top = `${top}px`;
}
export function hideTip() {
  tooltip().hidden = true;
}

function niceTicks(lo: number, hi: number, count: number): number[] {
  const span = hi - lo;
  const step0 = span / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((s) => s * mag).find((s) => s >= step0) ?? step0;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

/** Re-render on container resize (debounced to animation frames). */
export function responsive(container: HTMLElement, draw: (w: number) => void) {
  let last = 0;
  let raf = 0;
  const run = () => {
    const w = Math.round(container.clientWidth);
    if (w && w !== last) {
      last = w;
      draw(w);
    }
  };
  new ResizeObserver(() => {
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(run);
  }).observe(container);
  run();
  return () => {
    last = 0;
    run();
  };
}

export interface LineSeries {
  label: string;
  color: string; // css color / var()
  values: (number | null)[];
  width?: number;
  points?: boolean; // draw markers
  directLabel?: boolean;
  connect?: boolean; // bridge null gaps (sparse forecast points)
}
export interface Band {
  label: string;
  lo: (number | null)[];
  hi: (number | null)[];
  fill: string;
}
export interface LineChartOpts {
  months: Month[];
  series: LineSeries[];
  bands?: Band[];
  diverging?: (number | null)[]; // filled area above/below zero
  height: number;
  yDomain?: [number, number];
  title: string;
  desc: string;
  divider?: { at: Month; label: string };
  tooltipExtra?: (i: number) => string;
}

export function lineChart(container: HTMLElement, o: LineChartOpts) {
  return responsive(container, (W) => {
    container.replaceChildren();
    const H = o.height;
    const m = { t: 12, r: W < 520 ? 14 : 70, b: 26, l: 34 };
    const iw = W - m.l - m.r;
    const ih = H - m.t - m.b;
    const xs = o.months.map(monthToX);
    const x0 = xs[0];
    const x1 = xs[xs.length - 1];
    const all = [
      ...o.series.flatMap((s) => s.values),
      ...(o.bands ?? []).flatMap((b) => [...b.lo, ...b.hi]),
      ...(o.diverging ?? []),
    ].filter((v): v is number => v != null);
    let [ylo, yhi] = o.yDomain ?? [Math.min(-2.2, ...all) - 0.15, Math.max(2.2, ...all) + 0.15];
    ylo = Math.floor(ylo * 2) / 2;
    yhi = Math.ceil(yhi * 2) / 2;
    const X = (v: number) => m.l + ((v - x0) / (x1 - x0 || 1)) * iw;
    const Y = (v: number) => m.t + ((yhi - v) / (yhi - ylo)) * ih;

    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", tabindex: 0, "aria-labelledby": "" }, container);
    const id = `c${++uid}`;
    el("title", { id: `${id}-t` }, svg).textContent = o.title;
    el("desc", { id: `${id}-d` }, svg).textContent = o.desc + " Use the left and right arrow keys to read values.";
    svg.setAttribute("aria-labelledby", `${id}-t ${id}-d`);

    // grid + y axis
    const g = el("g", { class: "axis" }, svg);
    for (const t of niceTicks(ylo, yhi, H < 260 ? 4 : 6)) {
      el("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: t === 0 ? "baseline" : "gridline" }, g);
      el("text", { x: m.l - 6, y: Y(t) + 3.5, "text-anchor": "end" }, g).textContent = fmt(t, Number.isInteger(t) ? 0 : 1);
    }
    // x axis: year ticks
    const span = x1 - x0;
    const yStep = span > 40 ? 10 : span > 15 ? 5 : span > 6 ? 2 : span > 2.5 ? 1 : 0.5;
    const maxTicks = Math.max(2, Math.floor(iw / 56));
    let step = yStep;
    while (span / step > maxTicks) step *= 2;
    for (let v = Math.ceil(x0 / step) * step; v <= x1 + 1e-9; v += step) {
      const label = Number.isInteger(v) ? String(v) : `${MONTHS[Math.round((v % 1) * 12)]} ${Math.floor(v)}`;
      el("text", { x: X(v), y: H - 6, "text-anchor": "middle" }, g).textContent = label;
    }
    // drought threshold
    if (ylo < -1) {
      el("line", { x1: m.l, x2: W - m.r, y1: Y(-1), y2: Y(-1), class: "threshold" }, svg);
      el("text", { x: m.l + 4, y: Y(-1) - 4, class: "threshold-label" }, svg).textContent = "Drought threshold (−1)";
    }
    if (o.divider) {
      const dx = X(monthToX(o.divider.at));
      el("line", { x1: dx, x2: dx, y1: m.t, y2: H - m.b, class: "baseline" }, svg);
      el("text", { x: dx + 5, y: m.t + 10, class: "threshold-label" }, svg).textContent = o.divider.label;
    }

    const path = (vals: (number | null)[], connect = false) => {
      let d = "";
      let pen = false;
      vals.forEach((v, i) => {
        if (v == null) return void (pen = pen && connect);
        d += `${pen ? "L" : "M"}${X(xs[i]).toFixed(1)},${Y(v).toFixed(1)}`;
        pen = true;
      });
      return d;
    };

    if (o.diverging) {
      const v = o.diverging;
      const z = Y(0);
      let d = "";
      let start = -1;
      const flush = (end: number) => {
        if (start < 0) return;
        d += `M${X(xs[start]).toFixed(1)},${z}`;
        for (let i = start; i <= end; i++) d += `L${X(xs[i]).toFixed(1)},${Y(v[i] as number).toFixed(1)}`;
        d += `L${X(xs[end]).toFixed(1)},${z}Z`;
        start = -1;
      };
      v.forEach((val, i) => {
        if (val == null) flush(i - 1);
        else if (start < 0) start = i;
      });
      flush(v.length - 1);
      const defs = el("defs", {}, svg);
      const up = el("clipPath", { id: `${id}-up` }, defs);
      el("rect", { x: 0, y: 0, width: W, height: z }, up);
      const dn = el("clipPath", { id: `${id}-dn` }, defs);
      el("rect", { x: 0, y: z, width: W, height: H }, dn);
      el("path", { d, fill: "var(--wet-fill)", "clip-path": `url(#${id}-up)` }, svg);
      el("path", { d, fill: "var(--dry-fill)", "clip-path": `url(#${id}-dn)` }, svg);
    }

    for (const b of o.bands ?? []) {
      const idx = b.lo.map((v, i) => (v == null ? -1 : i)).filter((i) => i >= 0);
      if (!idx.length) continue;
      let d = idx.map((i, k) => `${k ? "L" : "M"}${X(xs[i]).toFixed(1)},${Y(b.hi[i] as number).toFixed(1)}`).join("");
      d += idx.reverse().map((i) => `L${X(xs[i]).toFixed(1)},${Y(b.lo[i] as number).toFixed(1)}`).join("") + "Z";
      el("path", { d, fill: b.fill }, svg);
    }
    for (const s of o.series) {
      el("path", { d: path(s.values, s.connect), fill: "none", stroke: s.color, "stroke-width": s.width ?? 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
      if (s.points)
        s.values.forEach((v, i) => {
          if (v != null) el("circle", { cx: X(xs[i]), cy: Y(v), r: 4.5, fill: s.color, stroke: "var(--surface)", "stroke-width": 2 }, svg);
        });
      if (s.directLabel && W >= 520) {
        let li = s.values.length - 1;
        while (li >= 0 && s.values[li] == null) li--;
        if (li >= 0)
          el("text", { x: X(xs[li]) + 8, y: Y(s.values[li] as number) + 4, class: "direct-label" }, svg).textContent = s.label;
      }
    }

    // hover layer
    const cross = el("line", { y1: m.t, y2: H - m.b, class: "crosshair", visibility: "hidden" }, svg);
    const dots = o.series.map((s) => el("circle", { r: 4.5, fill: s.color, stroke: "var(--surface)", "stroke-width": 2, visibility: "hidden" }, svg));
    let cur = -1;
    const show = (i: number, cx?: number, cy?: number) => {
      cur = Math.max(0, Math.min(xs.length - 1, i));
      const px = X(xs[cur]);
      cross.setAttribute("x1", String(px));
      cross.setAttribute("x2", String(px));
      cross.setAttribute("visibility", "visible");
      let rows = "";
      o.series.forEach((s, k) => {
        const v = s.values[cur];
        if (v == null) return void dots[k].setAttribute("visibility", "hidden");
        dots[k].setAttribute("cx", String(px));
        dots[k].setAttribute("cy", String(Y(v)));
        dots[k].setAttribute("visibility", "visible");
        rows += `<div class="row"><span><i class="swatch dot" style="background:${s.color}"></i>${s.label}</span><span>${fmt(v)}</span></div>`;
      });
      for (const b of o.bands ?? [])
        if (b.lo[cur] != null) rows += `<div class="row"><span>${b.label}</span><span>${fmt(b.lo[cur])} to ${fmt(b.hi[cur])}</span></div>`;
      if (o.diverging && o.diverging[cur] != null && !o.series.length)
        rows += `<div class="row"><span>Value</span><span>${fmt(o.diverging[cur])}</span></div>`;
      rows += o.tooltipExtra?.(cur) ?? "";
      const box = svg.getBoundingClientRect();
      const scale = box.width / W;
      showTip(`<strong>${fmtMonth(o.months[cur])}</strong>${rows}`, cx ?? box.left + px * scale, cy ?? box.top + m.t * scale + 40);
    };
    const hide = () => {
      cross.setAttribute("visibility", "hidden");
      dots.forEach((d) => d.setAttribute("visibility", "hidden"));
      hideTip();
    };
    svg.addEventListener("pointermove", (e) => {
      const box = svg.getBoundingClientRect();
      const px = ((e.clientX - box.left) / box.width) * W;
      const xv = x0 + ((px - m.l) / iw) * (x1 - x0);
      let best = 0;
      xs.forEach((v, i) => {
        if (Math.abs(v - xv) < Math.abs(xs[best] - xv)) best = i;
      });
      show(best, e.clientX, e.clientY);
    });
    svg.addEventListener("pointerleave", hide);
    svg.addEventListener("blur", hide);
    svg.addEventListener("keydown", (e) => {
      const k = e.key;
      if (k === "ArrowRight" || k === "ArrowLeft" || k === "Home" || k === "End") {
        e.preventDefault();
        const n = xs.length - 1;
        const next = k === "Home" ? 0 : k === "End" ? n : (cur < 0 ? n : cur) + (k === "ArrowRight" ? 1 : -1) * (e.shiftKey ? 12 : 1);
        show(next);
      } else if (k === "Escape") hide();
    });
  });
}

export interface BarItem {
  label: string;
  values: number[];
}
export function barChart(
  container: HTMLElement,
  o: { items: BarItem[]; series: { label: string; color: string }[]; title: string; desc: string; format?: (v: number) => string },
) {
  const f = o.format ?? ((v: number) => fmt(v, 3));
  return responsive(container, (W) => {
    container.replaceChildren();
    const nS = o.series.length;
    const barH = 12;
    const gap = 2;
    const groupH = nS * barH + (nS - 1) * gap;
    // Narrow screens: label sits above its bars instead of beside them.
    const stacked = W < 480;
    const labelH = stacked ? 18 : 0;
    const rowGap = nS > 1 ? 16 : 10;
    const labelW = stacked ? 0 : Math.min(230, Math.max(120, W * 0.42));
    const m = { t: 4, r: 52, b: 4, l: stacked ? 1 : labelW };
    const rowH = labelH + groupH;
    const H = m.t + m.b + o.items.length * (rowH + rowGap) - rowGap;
    const max = Math.max(...o.items.flatMap((d) => d.values), 1e-9);
    const X = (v: number) => (v / max) * (W - m.l - m.r);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, container);
    const id = `c${++uid}`;
    el("title", { id: `${id}-t` }, svg).textContent = o.title;
    el("desc", { id: `${id}-d` }, svg).textContent =
      o.desc + " " + o.items.map((d) => `${d.label}: ${d.values.map((v, k) => `${o.series[k].label} ${f(v)}`).join(", ")}`).join("; ");
    svg.setAttribute("aria-labelledby", `${id}-t ${id}-d`);
    o.items.forEach((d, i) => {
      const top = m.t + i * (rowH + rowGap);
      const y0 = top + labelH;
      const t = stacked
        ? el("text", { x: 2, y: top + 12, class: "tick" }, svg)
        : el("text", { x: m.l - 10, y: y0 + groupH / 2 + 4, "text-anchor": "end", class: "tick" }, svg);
      t.textContent = d.label;
      t.style.fill = "var(--ink-2)";
      t.style.fontSize = "12px";
      d.values.forEach((v, k) => {
        const y = y0 + k * (barH + gap);
        const w = Math.max(2, X(v));
        // 4px rounded data-end, square at the baseline
        const r = Math.min(4, w / 2);
        const p = `M${m.l},${y}h${w - r}a${r},${r} 0 0 1 ${r},${r}v${barH - 2 * r}a${r},${r} 0 0 1 ${-r},${r}h${-(w - r)}Z`;
        const bar = el("path", { d: p, fill: o.series[k].color }, svg);
        const hit = el("rect", { x: m.l, y: y - gap / 2, width: W - m.l - m.r, height: barH + gap, fill: "transparent" }, svg);
        const tip = (e: PointerEvent) =>
          showTip(`<strong>${d.label}</strong><div class="row"><span><i class="swatch dot" style="background:${o.series[k].color}"></i>${o.series[k].label}</span><span>${f(v)}</span></div>`, e.clientX, e.clientY);
        hit.addEventListener("pointermove", tip);
        hit.addEventListener("pointerleave", hideTip);
        void bar;
        if (k === 0 || nS === 1) {
          const lab = el("text", { x: m.l + w + 6, y: y + barH - 2, class: "tick" }, svg);
          lab.textContent = f(v);
        }
      });
    });
    el("line", { x1: m.l, x2: m.l, y1: 0, y2: H, class: "baseline" }, svg);
  });
}

export const CLASS_BINS = [
  { max: -2, label: "Extreme drought", cls: "c0" },
  { max: -1.5, label: "Severe drought", cls: "c1" },
  { max: -1, label: "Moderate drought", cls: "c2" },
  { max: 1, label: "Near normal", cls: "c3" },
  { max: 1.5, label: "Moderately wet", cls: "c4" },
  { max: 2, label: "Very wet", cls: "c5" },
  { max: Infinity, label: "Extremely wet", cls: "c6" },
];
// Mirrors spei_class() in pipeline/train.py.
export const binOf = (v: number) =>
  CLASS_BINS[v <= -2 ? 0 : v <= -1.5 ? 1 : v <= -1 ? 2 : v < 1 ? 3 : v < 1.5 ? 4 : v < 2 ? 5 : 6];

export function calendarHeatmap(container: HTMLElement, months: Month[], values: (number | null)[], label: string) {
  return responsive(container, (W0) => {
    container.replaceChildren();
    const W = Math.max(520, W0);
    const years = [...new Set(months.map((m) => +m.slice(0, 4)))];
    const m = { t: 4, r: 4, b: 22, l: 30 };
    const cw = (W - m.l - m.r) / years.length;
    const ch = Math.max(10, Math.min(18, cw * 1.25));
    const H = m.t + m.b + ch * 12;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, container);
    const id = `c${++uid}`;
    el("title", { id: `${id}-t` }, svg).textContent = `${label} drought calendar, ${years[0]}–${years[years.length - 1]}`;
    const nDrought = values.filter((v) => v != null && v <= -1).length;
    el("desc", { id: `${id}-d` }, svg).textContent =
      `Each cell is one month coloured by drought class. ${nDrought} of ${values.filter((v) => v != null).length} months were at or below −1. A yearly table is available below the history chart.`;
    svg.setAttribute("aria-labelledby", `${id}-t ${id}-d`);
    ["J", "", "M", "", "M", "", "J", "", "S", "", "N", ""].forEach((t, i) => {
      if (t) el("text", { x: m.l - 8, y: m.t + i * ch + ch / 2 + 4, "text-anchor": "end", class: "tick" }, svg).textContent = MONTHS[i].slice(0, 3);
    });
    const step = cw < 9 ? 10 : 5;
    years.forEach((y, i) => {
      if (y % step === 0) el("text", { x: m.l + i * cw + cw / 2, y: H - 6, "text-anchor": "middle", class: "tick" }, svg).textContent = String(y);
    });
    const g = el("g", {}, svg);
    months.forEach((mo, k) => {
      const v = values[k];
      if (v == null) return;
      const yi = years.indexOf(+mo.slice(0, 4));
      const mi = +mo.slice(5, 7) - 1;
      const r = el("rect", {
        x: (m.l + yi * cw + 0.5).toFixed(2),
        y: (m.t + mi * ch + 0.5).toFixed(2),
        width: Math.max(1, cw - 1).toFixed(2),
        height: (ch - 1).toFixed(2),
        rx: 1.5,
        class: `cell ${binOf(v).cls}`,
      }, g);
      r.dataset.k = String(k);
    });
    g.addEventListener("pointermove", (e) => {
      const t = e.target as SVGElement;
      if (!t.dataset?.k) return hideTip();
      const k = +t.dataset.k;
      const v = values[k] as number;
      showTip(`<strong>${fmtMonth(months[k])}</strong><div class="row"><span>${label}</span><span>${fmt(v)}</span></div><div>${binOf(v).label}</div>`, e.clientX, e.clientY);
    });
    g.addEventListener("pointerleave", hideTip);
  });
}
