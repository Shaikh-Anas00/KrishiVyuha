/**
 * RotationPanel — multi-season, rotation-aware plan vs. static / greedy baselines.
 */
"use client";
import React from "react";
import type { RotationResponse } from "@/lib/api";

const INR = (n: number) =>
  new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(n);

const top = (w: Record<string, number>) =>
  Object.entries(w).filter(([, v]) => v > 0.01).sort(([, a], [, b]) => b - a)
    .map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`).join(" · ");

export default function RotationPanel({ data }: { data: RotationResponse }) {
  const s = data.summary;
  return (
    <div className="rounded-2xl bg-white ring-1 ring-gray-200 shadow-sm p-6 space-y-4">
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-lime-100 flex items-center justify-center text-lg">🔄</div>
        <div>
          <h3 className="font-bold text-gray-900">{data.n_seasons}-Season Rotation Plan</h3>
          <p className="text-xs text-gray-500">Crop-interaction matrix · legume N-credit · monoculture penalty</p>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm">
        {[
          { label: "Static plan repeated", v: s.static_discounted_profit, tone: "bg-gray-50" },
          { label: "Greedy (season-by-season)", v: s.greedy_discounted_profit, tone: "bg-gray-50" },
          { label: "Rotation-aware", v: s.rotation_discounted_profit, tone: "bg-emerald-50 ring-1 ring-emerald-200" },
        ].map((c) => (
          <div key={c.label} className={`rounded-xl p-3 ${c.tone}`}>
            <p className="text-xs text-gray-500">{c.label}</p>
            <p className="font-bold text-gray-900">{INR(c.v)}</p>
          </div>
        ))}
      </div>
      <p className="text-sm text-emerald-700 font-medium">
        +{s.gain_vs_static_pct}% vs static repeat ({INR(s.gain_vs_static_inr)}) · +{s.gain_vs_greedy_pct}% vs greedy ({INR(s.gain_vs_greedy_inr)})
        <span className="text-xs text-gray-400 font-normal"> (discounted expected profit)</span>
      </p>

      <div className="space-y-2">
        {data.rotation_plan.map((p) => (
          <div key={p.season_index} className="rounded-xl bg-slate-50 border border-slate-200 p-3 text-xs space-y-1">
            <div className="flex justify-between">
              <span className="font-bold text-gray-900">Season {p.season_index}</span>
              <span className="text-gray-500">Profit {INR(p.expected_profit)} · Worst-case {INR(p.worst_case_income)}</span>
            </div>
            <p className="text-gray-700">{top(p.weights)}</p>
            <p className="text-gray-500">
              Plot moves: {p.plot_transitions.map((t) => `${t.from}→${t.to} ${(t.share * 100).toFixed(0)}%`).join(", ")}
            </p>
          </div>
        ))}
      </div>
      <p className="text-[11px] text-gray-400">
        Legume share per season — rotation: {s.legume_share_rotation.map((x) => `${(x * 100).toFixed(0)}%`).join(" / ")}
        {" "}| static: {s.legume_share_static.map((x) => `${(x * 100).toFixed(0)}%`).join(" / ")}
      </p>
      <p className="text-[11px] text-amber-700 bg-amber-50 rounded-lg p-2">⚠ {data.assumption_note}</p>
    </div>
  );
}
