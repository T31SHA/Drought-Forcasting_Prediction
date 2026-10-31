// Pure string renderers shared by the browser bundle and vite.config.ts, which
// pre-renders the hero cards into index.html at build time (no layout shift,
// content visible before JavaScript runs). Must not touch the DOM.
import type { Payload } from "./types";
import { fmt, fmtMonth } from "./charts";

export const pct = (v: number | null | undefined, nd = 0) => (v == null ? "—" : `${(v * 100).toFixed(nd)}%`);
// 0 historical errors reaching −1 means "under 1 in ~500", not "impossible".
export const pctProb = (v: number) => (v < 0.01 ? "< 1%" : pct(v));
export const hLabel = (h: number) => `${h} month${h > 1 ? "s" : ""}`;

// Risk levels, driven by the conformal probability that SPEI-12 ends at or below −1.
const RISK = [
  { min: 0.5, key: "high", label: "High drought risk", icon: '<path d="M8 1.5 14.5 8 8 14.5 1.5 8Z" fill="currentColor"/><path d="M8 4.8v4" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/><circle cx="8" cy="11.3" r="1" fill="#fff"/>' },
  { min: 0.25, key: "warning", label: "Elevated risk", icon: '<path d="M8 1.6 15 14H1Z" fill="currentColor"/><path d="M8 6v3.8" stroke="#0b0b0b" stroke-width="1.6" stroke-linecap="round"/><circle cx="8" cy="11.8" r=".9" fill="#0b0b0b"/>' },
  { min: 0.1, key: "watch", label: "Watch", icon: '<circle cx="8" cy="8" r="6.5" fill="currentColor"/><path d="M8 4.5v4" stroke="#0b0b0b" stroke-width="1.6" stroke-linecap="round"/><circle cx="8" cy="11" r=".9" fill="#0b0b0b"/>' },
  { min: 0, key: "low", label: "Low drought risk", icon: '<circle cx="8" cy="8" r="6.5" fill="currentColor"/><path d="m5 8.2 2 2 4-4.2" fill="none" stroke="#fff" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>' },
];
export const riskBadge = (p: number) => {
  const r = RISK.find((x) => p >= x.min)!;
  return `<span class="risk risk-${r.key}"><svg aria-hidden="true" width="16" height="16" viewBox="0 0 16 16">${r.icon}</svg>${r.label}</span>`;
};

export function cardsHtml(data: Payload): string {
  const { current, forecast } = data;
  const now = `<article class="card now" aria-label="Current conditions">
      <span class="card-label">Now</span>
      <span class="card-date">${fmtMonth(current.date)} · observed</span>
      <span class="card-value">${fmt(current.spei12)}</span>
      <span class="card-class">${current.class}</span>
      <span class="card-range">SPEI-12, standard score</span>
    </article>`;
  const cards = forecast
    .map((f) => {
      const [lo, hi] = f.intervals["80"];
      return `<article class="card" aria-label="Forecast ${hLabel(f.horizon)} ahead">
        <span class="card-label">+${hLabel(f.horizon)}</span>
        <span class="card-date">${fmtMonth(f.target_date)}</span>
        <span class="card-value">${fmt(f.prediction)}</span>
        <span class="card-class">${f.class}</span>
        <span class="card-range">Likely range ${fmt(lo)} to ${fmt(hi)}</span>
        ${riskBadge(f.p_drought)}
        <span class="prob">Chance of drought (≤ −1): ${pctProb(f.p_drought)}</span>
      </article>`;
    })
    .join("");
  return now + cards;
}

export const forecastSub = (data: Payload) =>
  `Issued from the latest observed month, ${fmtMonth(data.current.date)}. Ranges are 80% prediction intervals.`;
export const dataRange = (data: Payload) => `Record: ${fmtMonth(data.meta.data_start)} – ${fmtMonth(data.meta.data_end)}.`;

/** Replace <!--ssr:key--> markers in index.html. */
export function prerender(html: string, data: Payload): string {
  const parts: Record<string, string> = { cards: cardsHtml(data), sub: forecastSub(data), range: dataRange(data) };
  return html.replace(/<!--ssr:(\w+)-->/g, (_, k: string) => parts[k] ?? "");
}
