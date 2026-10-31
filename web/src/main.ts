import "./style.css";
import raw from "./data/forecast.json";
import type { HorizonForecast, Metric, Payload } from "./types";
import { CLASS_BINS, barChart, binOf, calendarHeatmap, fmt, fmtMonth, fmtSigned, lineChart, type Month } from "./charts";
import { hLabel, pct, pctProb } from "./static";

const data = raw as unknown as Payload;
const $ = (id: string) => document.getElementById(id) as HTMLElement;
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
const addMonths = (m: Month, k: number): Month => {
  const [y, mo] = m.split("-").map(Number);
  const t = y * 12 + (mo - 1) + k;
  return `${Math.floor(t / 12)}-${String((t % 12) + 1).padStart(2, "0")}`;
};

// ---- Forecast ------------------------------------------------------------
function renderForecast() {
  const { current, forecast } = data;

  // Outlook chart: last 36 observed months + forecast fan.
  const obsDates = data.observed.dates;
  const n = obsDates.length;
  const start = n - 36;
  const last = obsDates[n - 1];
  const maxH = Math.max(...forecast.map((f) => f.horizon));
  const months = [...obsDates.slice(start), ...Array.from({ length: maxH }, (_, i) => addMonths(last, i + 1))];
  const observed = months.map((_, i) => (i < 36 ? data.observed.spei12[start + i] : null));
  const fc: (number | null)[] = months.map(() => null);
  const lo80 = [...fc], hi80 = [...fc], lo95 = [...fc], hi95 = [...fc];
  fc[35] = lo80[35] = hi80[35] = lo95[35] = hi95[35] = current.spei12;
  for (const f of forecast) {
    const i = 35 + f.horizon;
    fc[i] = f.prediction;
    [lo80[i], hi80[i]] = f.intervals["80"];
    [lo95[i], hi95[i]] = f.intervals["95"];
  }
  lineChart($("chart-outlook"), {
    months,
    height: 300,
    title: "Recent SPEI-12 and forecast",
    desc: `Observed SPEI-12 from ${fmtMonth(months[0])} to ${fmtMonth(last)}, then forecasts: ${forecast
      .map((f) => `${fmtMonth(f.target_date)} ${fmt(f.prediction)} (80% range ${fmt(f.intervals["80"][0])} to ${fmt(f.intervals["80"][1])})`)
      .join("; ")}.`,
    bands: [
      { label: "95% interval", lo: lo95, hi: hi95, fill: "var(--band-95)" },
      { label: "80% interval", lo: lo80, hi: hi80, fill: "var(--band-80)" },
    ],
    series: [
      { label: "Observed", color: "var(--s-observed)", values: observed },
      { label: "Forecast", color: "var(--s-forecast)", values: fc, points: true, connect: true, directLabel: true },
    ],
    divider: { at: last, label: "Latest data" },
  });
  $("legend-outlook").innerHTML = `
    <span><i class="swatch" style="background:var(--s-observed)"></i>Observed SPEI-12</span>
    <span><i class="swatch dot" style="background:var(--s-forecast)"></i>Forecast</span>
    <span><i class="swatch box" style="background:var(--band-80)"></i>80% interval</span>
    <span><i class="swatch box" style="background:var(--band-95)"></i>95% interval</span>`;
  $("table-outlook").innerHTML = `<table><caption class="sr-only">Forecast details</caption><thead><tr>
      <th scope="col">Target month</th><th scope="col">Lead</th><th scope="col">Method</th><th scope="col">SPEI-12</th><th scope="col">Class</th>
      <th scope="col">80% interval</th><th scope="col">95% interval</th><th scope="col">P(drought)</th></tr></thead><tbody>
      ${forecast
        .map(
          (f) => `<tr><td>${fmtMonth(f.target_date)}</td><td>${hLabel(f.horizon)}</td><td>${f.model}</td><td>${fmt(f.prediction)}</td><td>${f.class}</td>
          <td>${fmt(f.intervals["80"][0])} to ${fmt(f.intervals["80"][1])}</td><td>${fmt(f.intervals["95"][0])} to ${fmt(f.intervals["95"][1])}</td><td>${pctProb(f.p_drought)}</td></tr>`,
        )
        .join("")}</tbody></table>`;
  renderWhy(forecast);
}

function renderWhy(forecast: HorizonForecast[]) {
  const cur = data.current;
  const parts = forecast.map((f) => {
    const delta = f.prediction - cur.spei12;
    const body = f.drivers.length
      ? `<ol>${f.drivers
          .slice(0, 4)
          .map((d) => `<li><span class="up">${esc(d.label)}</span> (${fmt(d.value)}) ${d.contribution >= 0 ? "pushes up" : "pulls down"} ${fmt(Math.abs(d.contribution))}</li>`)
          .join("")}</ol>`
      : `<p>The ${f.model.toLowerCase()} baseline won cross-validation here, so no learned features are used.</p>`;
    return `<div><h4>${fmtMonth(f.target_date)} · ${f.model} · change ${fmtSigned(delta)}</h4>${body}</div>`;
  });
  $("why").innerHTML = `<h3>Why the model expects this</h3>
    <p>Each forecast starts from today's SPEI-12 (${fmt(cur.spei12)}) and adds a learned change. The strongest contributions to that change are listed below (SHAP values, in SPEI units, relative to the model's average). Contributions describe the model's arithmetic, not physical causes.</p>
    <div class="drivers">${parts.join("")}</div>`;
}

// ---- History -------------------------------------------------------------
const SCALES = [
  { key: "spei1", label: "SPEI-1" },
  { key: "spei3", label: "SPEI-3" },
  { key: "spei6", label: "SPEI-6" },
  { key: "spei12", label: "SPEI-12" },
  { key: "spei24", label: "SPEI-24" },
] as const;
const RANGES = [
  { years: 10, label: "10 yrs" },
  { years: 30, label: "30 yrs" },
  { years: 0, label: "All" },
];
let scale: (typeof SCALES)[number] = SCALES[3];
let range = RANGES[2];

function segButtons(host: HTMLElement, items: readonly { label: string }[], isOn: (i: number) => boolean, onPick: (i: number) => void) {
  host.innerHTML = items.map((it, i) => `<button type="button" aria-pressed="${isOn(i)}">${it.label}</button>`).join("");
  host.querySelectorAll("button").forEach((b, i) =>
    b.addEventListener("click", () => {
      onPick(i);
      host.querySelectorAll("button").forEach((x, k) => x.setAttribute("aria-pressed", String(isOn(k))));
    }),
  );
}

function renderHistory() {
  const dates = data.observed.dates;
  const vals = data.observed[scale.key];
  const from = range.years ? dates.length - range.years * 12 : 0;
  const months = dates.slice(from);
  const v = vals.slice(from);
  const first = v.findIndex((x) => x != null);
  $("history-chart-title").textContent = `${scale.label}, monthly`;
  const drought = v.filter((x) => x != null && x <= -1).length;
  lineChart($("chart-history"), {
    months: months.slice(first),
    height: 320,
    title: `${scale.label} history`,
    desc: `${scale.label} from ${fmtMonth(months[first])} to ${fmtMonth(months[months.length - 1])}. ${drought} months at or below the drought threshold of −1.`,
    diverging: v.slice(first),
    series: [{ label: scale.label, color: "var(--ink-2)", values: v.slice(first), width: 1 }],
    tooltipExtra: (i) => {
      const x = v[first + i];
      return x == null ? "" : `<div>${binOf(x).label}</div>`;
    },
  });
  calendarHeatmap($("chart-calendar"), dates, vals, scale.label);

  // yearly table (non-visual view of history)
  const byYear = new Map<number, number[]>();
  dates.forEach((d, i) => {
    const x = vals[i];
    if (x == null) return;
    const y = +d.slice(0, 4);
    (byYear.get(y) ?? byYear.set(y, []).get(y)!).push(x);
  });
  const rows = [...byYear.entries()]
    .reverse()
    .map(([y, xs]) => `<tr><td>${y}</td><td>${fmt(xs.reduce((a, b) => a + b, 0) / xs.length)}</td><td>${fmt(Math.min(...xs))}</td><td>${xs.filter((x) => x <= -1).length}</td></tr>`)
    .join("");
  $("table-history").innerHTML = `<table><thead><tr><th scope="col">Year</th><th scope="col">Mean ${scale.label}</th><th scope="col">Lowest</th><th scope="col">Drought months</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function initHistory() {
  segButtons($("scale-buttons"), SCALES, (i) => SCALES[i] === scale, (i) => ((scale = SCALES[i]), renderHistory()));
  segButtons($("range-buttons"), RANGES, (i) => RANGES[i] === range, (i) => ((range = RANGES[i]), renderHistory()));
  $("legend-calendar").innerHTML = CLASS_BINS.map((b) => `<span><i class="swatch box ${b.cls}"></i>${b.label}</span>`).join("");
  renderHistory();
}

// ---- Performance ---------------------------------------------------------
const TREND = " +trend";
const MODEL_ORDER = ["Persistence", "Climatology", "Damped persistence"]
  .concat(...["Ridge", "XGBoost", "Ridge + XGBoost"].map((m) => [m, m + TREND]))
  .concat("Previous Ridge");
const MODEL_NOTE: Record<string, string> = {
  Persistence: "next = today",
  Climatology: "monthly average",
  "Damped persistence": "a + b·today",
  Ridge: "linear, 37 features",
  XGBoost: "boosted trees, 60 Optuna trials",
  "Ridge + XGBoost": "equal-weight average",
  "Previous Ridge": "before this update · original code",
};
const modelNote = (m: string) =>
  m.endsWith(TREND) ? `${MODEL_NOTE[m.slice(0, -TREND.length)]} + year/decade trend` : (MODEL_NOTE[m] ?? "");

function renderPerformance(h: number) {
  const bt = data.backtest[String(h)];
  const sel = bt.selected;
  const rows = data.metrics.filter((m) => m.horizon === h);
  const s = rows.find((m) => m.model === sel)!;
  const pers = rows.find((m) => m.model === "Persistence")!;
  const prev = rows.find((m) => m.model === "Previous Ridge");
  // Coverage of the intervals actually drawn (adaptive or static).
  const btPred = bt.predictions[sel];
  const shownCoverage = bt.observed.filter((o, i) => Math.abs(o - btPred[i]) <= bt.radius["80"][i]).length / bt.observed.length;
  const boot = data.bootstrap[String(h)];
  const skillCls = s.skill_vs_persistence > 0 ? "delta-good" : s.skill_vs_persistence < 0 ? "delta-bad" : "";

  $("perf-stats").innerHTML = `
    <div class="stat"><div class="stat-label">Selected method</div><div class="stat-value">${sel}</div><div class="stat-note">chosen by cross-validation</div></div>
    <div class="stat"><div class="stat-label">Typical error (RMSE)</div><div class="stat-value">${fmt(s.rmse, 3)}</div><div class="stat-note">persistence ${fmt(pers.rmse, 3)}</div></div>
    <div class="stat"><div class="stat-label">Skill vs persistence</div><div class="stat-value ${skillCls}">${fmtSigned(s.skill_vs_persistence * 100, 0)}%</div><div class="stat-note">${s.skill_vs_persistence >= 0 ? "less squared error than persistence" : "more squared error than persistence"}</div></div>
    ${prev ? `<div class="stat"><div class="stat-label">vs previous version</div><div class="stat-value ${prev.rmse < s.rmse ? "delta-bad" : "delta-good"}">${fmt(prev.rmse, 3)}</div><div class="stat-note">${prev.rmse < s.rmse ? `previous Ridge was better here (lower RMSE in ${pct(boot.p_lower[`Previous Ridge < ${sel}`] ?? null)} of bootstrap resamples)` : "previous Ridge was worse here"}</div></div>` : ""}
    <div class="stat"><div class="stat-label">80% interval coverage</div><div class="stat-value">${pct(shownCoverage)}</div><div class="stat-note">target 80%${bt.interval_method === "adaptive conformal" ? " · adaptive range" : ""} · R² ${fmt(s.r2, 2)}</div></div>`;

  $("backtest-h").textContent = hLabel(h);
  $("backtest-title").textContent = `Backtest, ${hLabel(h)} ahead`;
  const pred = bt.predictions[sel];
  const r80 = bt.radius["80"];
  const adaptive = bt.interval_method === "adaptive conformal";
  lineChart($("chart-backtest"), {
    months: bt.dates,
    height: 300,
    title: `Backtest ${hLabel(h)} ahead, ${fmtMonth(bt.dates[0])} to ${fmtMonth(bt.dates[bt.dates.length - 1])}`,
    desc: `Observed SPEI-12 compared with ${sel} forecasts made ${hLabel(h)} earlier and the persistence baseline. RMSE ${fmt(s.rmse, 3)} versus ${fmt(pers.rmse, 3)} for persistence.`,
    bands: [{ label: "80% interval", lo: pred.map((p, i) => p - r80[i]), hi: pred.map((p, i) => p + r80[i]), fill: "var(--band-80)" }],
    series: [
      ...(sel === "Persistence" ? [] : [{ label: "Persistence", color: "var(--s-baseline)", values: bt.predictions["Persistence"], width: 1.5 }]),
      { label: "Observed", color: "var(--s-observed)", values: bt.observed },
      { label: sel, color: "var(--s-forecast)", values: pred },
    ],
  });
  $("legend-backtest").innerHTML = `
    <span><i class="swatch" style="background:var(--s-observed)"></i>Observed</span>
    <span><i class="swatch" style="background:var(--s-forecast)"></i>${sel} forecast</span>
    <span><i class="swatch box" style="background:var(--band-80)"></i>80% interval</span>
    ${sel === "Persistence" ? "" : `<span><i class="swatch" style="background:var(--s-baseline)"></i>Persistence</span>`}`;

  const sorted = MODEL_ORDER.map((name) => rows.find((m) => m.model === name)).filter(Boolean) as Metric[];
  const best = Math.min(...sorted.map((m) => m.rmse));
  const tr = (m: Metric) => `<tr class="${m.model === sel ? "selected" : ""}">
      <th scope="row">${m.model}${m.model === sel ? '<span class="tag">selected</span>' : ""}${m.rmse === best ? '<span class="tag">lowest RMSE</span>' : ""}<br><span class="muted" style="font-weight:400;font-size:.8rem">${modelNote(m.model)}</span></th>
      <td>${fmt(m.rmse, 3)}</td><td>${fmt(m.mae, 3)}</td><td>${fmt(m.r2, 3)}</td><td>${fmt(m.explained_variance, 3)}</td><td>${fmtSigned(m.bias, 3)}</td>
      <td class="${m.skill_vs_persistence > 0.0005 ? "delta-good" : m.skill_vs_persistence < -0.0005 ? "delta-bad" : ""}">${m.model === "Persistence" ? "0 (ref.)" : fmtSigned(m.skill_vs_persistence * 100, 1) + "%"}</td>
      <td>${m.model === sel && bt.interval_method === "adaptive conformal"
        ? `${pct(shownCoverage)} <span class="muted" style="font-weight:400">(adaptive; static ${pct(m.coverage_80)})</span>`
        : m.coverage_80 == null ? "—" : pct(m.coverage_80)}</td></tr>`;
  $("table-metrics").innerHTML = `<table><thead><tr><th scope="col">Method</th><th scope="col">RMSE</th><th scope="col">MAE</th><th scope="col">R²</th><th scope="col">Expl. var.</th><th scope="col">Bias</th><th scope="col">Skill vs persistence</th><th scope="col">80% PI coverage</th></tr></thead>
    <tbody>${sorted.map(tr).join("")}</tbody></table>
    <p class="fig-sub" style="margin:.75rem 0 0">Cross-validation over 1973–2015 (not shown here) ranked the methods differently: the new Ridge had the lowest error at every horizon, which is why it was selected. See the README for both tables.</p>`;

  const seasons = ["DJF (winter)", "MAM (spring)", "JJA (summer)", "SON (autumn)"];
  const sRow = (model: string, season: string) => data.season.find((x) => x.model === model && x.horizon === h && x.season === season);
  const compare = sel === "Persistence" ? "XGBoost" : "Persistence";
  $("backtest-interval-note").textContent = adaptive
    ? " The 80% range is adaptive: each month it is widened or narrowed using only outcomes already observed by then."
    : "";
  barChart($("chart-season"), {
    title: `RMSE by season, ${hLabel(h)} ahead`,
    desc: `Test-window RMSE of ${sel} and ${compare} by season of the target month.`,
    series: [
      { label: sel, color: "var(--s-forecast)" },
      { label: compare, color: "var(--s-baseline)" },
    ],
    items: seasons.map((se) => ({ label: se, values: [sRow(sel, se)?.rmse ?? 0, sRow(compare, se)?.rmse ?? 0] })),
  });
  $("legend-season").innerHTML = `<span><i class="swatch box" style="background:var(--s-forecast)"></i>${sel}</span><span><i class="swatch box" style="background:var(--s-baseline)"></i>${compare}</span>`;

  barChart($("chart-importance"), {
    title: `Feature importance, ${hLabel(h)} ahead`,
    desc: "Mean absolute SHAP value of the XGBoost model on the test window, in SPEI units.",
    series: [{ label: "Mean |SHAP|", color: "var(--s-observed)" }],
    items: data.importance[String(h)].map((d) => ({ label: d.label, values: [d.value] })),
  });
}

function initPerformance() {
  const tabs = $("horizon-tabs");
  const hs = data.forecast.map((f) => f.horizon);
  tabs.innerHTML = hs
    .map((h, i) => `<button type="button" role="tab" id="horizon-tab-${h}" aria-controls="perf-panel" aria-selected="${i === 0}" tabindex="${i === 0 ? 0 : -1}" aria-label="${hLabel(h)} ahead">${hLabel(h)}</button>`)
    .join("");
  const btns = [...tabs.querySelectorAll<HTMLButtonElement>("button")];
  const pick = (i: number, focus = false) => {
    btns.forEach((b, k) => {
      b.setAttribute("aria-selected", String(k === i));
      b.tabIndex = k === i ? 0 : -1;
    });
    if (focus) btns[i].focus();
    $("perf-panel").setAttribute("aria-labelledby", btns[i].id);
    renderPerformance(hs[i]);
  };
  btns.forEach((b, i) => {
    b.addEventListener("click", () => pick(i));
    b.addEventListener("keydown", (e) => {
      const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (d) {
        e.preventDefault();
        pick((i + d + btns.length) % btns.length, true);
      }
    });
  });
  pick(0);
}

// Render in separate tasks so no single long task blocks input: the hero cards
// are already in the HTML (pre-rendered at build time), charts follow.
const steps = [renderForecast, initHistory, initPerformance];
const next = () => {
  const step = steps.shift();
  if (!step) return;
  step();
  setTimeout(next, 0);
};
next();
