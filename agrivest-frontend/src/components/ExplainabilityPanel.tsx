/**
 * ExplainabilityPanel v2 — Agro-Climatic Advisory Engine Edition
 * Renders:
 *   - Optimizer status badge
 *   - Climate auto-detection info card
 *   - Per-crop agronomic diagnostic reasons (from weather_service)
 *   - Risk concentration analysis
 *   - Survival budget check
 *   - PMFBY payout breakdown
 */
"use client";
import React, { useState } from "react";
import type { PortfolioResponse } from "@/lib/api";

interface Props {
  result: PortfolioResponse;
  survivalBudget: number;
  totalAreaHa: number;
}

const INR = (n: number) =>
  new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(n);

const REGIME_BADGE: Record<string, { color: string; emoji: string; label: string }> = {
  El_Nino: { color: "bg-orange-100 text-orange-800 ring-orange-200", emoji: "🔴", label: "El Niño" },
  La_Nina: { color: "bg-blue-100   text-blue-800   ring-blue-200",   emoji: "🔵", label: "La Niña" },
  Normal:  { color: "bg-green-100  text-green-800  ring-green-200",  emoji: "🟢", label: "Normal"  },
};

export default function ExplainabilityPanel({ result, survivalBudget, totalAreaHa }: Props) {
  const [expandedCrop, setExpandedCrop] = useState<string | null>(null);

  const sortedCrops   = Object.entries(result.weights).sort(([, a], [, b]) => b - a);
  const dominantCrop  = sortedCrops[0];
  const secondaryCrop = sortedCrops[1];
  const dominantPct   = (dominantCrop[1] * 100).toFixed(1);
  const dominantArea  = result.area_ha[dominantCrop[0]]?.toFixed(2) ?? "–";
  const budgetMet     = result.worst_case_income >= survivalBudget;
  const netProfit     = result.expected_income - result.total_premium;

  const riskLevel =
    dominantCrop[1] > 0.7 ? "HIGH concentration"
    : dominantCrop[1] > 0.5 ? "MODERATE concentration"
    : "WELL DIVERSIFIED";
  const riskColor =
    dominantCrop[1] > 0.7 ? "text-rose-600"
    : dominantCrop[1] > 0.5 ? "text-amber-600"
    : "text-emerald-600";

  const pmfbyInsights = Object.entries(result.crop_details)
    .filter(([, d]) => d.pmfby_payout > 0)
    .map(([crop, d]) => ({ crop, payout: d.pmfby_payout, premium: d.premium_cost }));

  // Climate info
  const ci = result.climate_info || {};
  const regime = ci.climate_regime ?? "Normal";
  const regimeBadge = REGIME_BADGE[regime] ?? REGIME_BADGE["Normal"];

  return (
    <div className="rounded-2xl bg-white ring-1 ring-gray-200 shadow-sm p-6 space-y-5">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-violet-100 flex items-center justify-center text-lg">🧠</div>
        <div>
          <h3 className="font-bold text-gray-900">AI Explainability & Advisory Summary</h3>
          <p className="text-xs text-gray-500">Auto-detected climate · Agronomic suitability · Risk analysis</p>
        </div>
      </div>

      {/* Optimizer status */}
      <div className={`inline-flex items-center gap-2 text-sm font-medium px-3 py-1.5 rounded-full
        ${result.success ? "bg-emerald-100 text-emerald-800" : "bg-amber-100 text-amber-800"}`}>
        <span>{result.success ? "✅" : "⚠️"}</span>
        {result.message}
      </div>

      {/* Proxied Crop Warning */}
      {result.proxied_crops && Object.keys(result.proxied_crops).length > 0 && (
        <div className="rounded-xl p-3.5 bg-amber-50 border border-amber-200 text-xs text-amber-900 space-y-1">
          <p className="font-semibold flex items-center gap-1.5 text-amber-800">
            <span>ℹ️</span> Proxied Horticultural Risk Profile Active
          </p>
          <p className="text-amber-700">
            The following crop(s) utilize benchmark-scaled proxies rather than dedicated historical census records:
          </p>
          <ul className="list-disc list-inside space-y-0.5 text-amber-800 font-mono text-[11px]">
            {Object.entries(result.proxied_crops).map(([crop, note]) => (
              <li key={crop}><strong>{crop}</strong>: {note}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Auto-Detected Climate Card */}
      {Object.keys(ci).length > 0 && (
        <div className={`rounded-xl p-4 ring-1 space-y-2 ${regimeBadge.color} ring-current`}>
          <div className="flex items-center justify-between">
            <p className="font-semibold text-sm flex items-center gap-2">
              <span>{regimeBadge.emoji}</span>
              Live Climate Sync: <span className="font-bold">{regimeBadge.label}</span> Detected
            </p>
            <span className="text-xs opacity-70">ONI = {ci.oni_value}</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="opacity-70">Adjusted Rainfall:</span>
              <span className="font-bold ml-1">{ci.rainfall_mm} mm</span>
              {ci.enso_rain_adj_pct !== 0 && (
                <span className="ml-1 text-xs opacity-70">
                  ({ci.enso_rain_adj_pct! > 0 ? "+" : ""}{ci.enso_rain_adj_pct}% from baseline {ci.baseline_rain} mm)
                </span>
              )}
            </div>
            <div>
              <span className="opacity-70">Temperature:</span>
              <span className="font-bold ml-1">{ci.temperature_c}°C</span>
              {ci.enso_temp_adj_c !== 0 && (
                <span className="ml-1 text-xs opacity-70">
                  ({ci.enso_temp_adj_c! > 0 ? "+" : ""}{ci.enso_temp_adj_c}°C from baseline)
                </span>
              )}
            </div>
          </div>
          <p className="text-xs opacity-70">
            Data source: {ci.baseline_source === "district+season"
              ? "District historical median (1997–2024)"
              : ci.baseline_source === "state+season"
              ? "State-level seasonal median (fallback)"
              : "National seasonal median (fallback)"}
          </p>
        </div>
      )}

      {/* Per-crop agronomic diagnostics */}
      {result.agronomic_diagnostics && Object.keys(result.agronomic_diagnostics).length > 0 && (
        <div className="space-y-2">
          <p className="text-sm font-semibold text-gray-700 flex items-center gap-2">
            <span>🌱</span> Agronomic Suitability Analysis
            <span className="text-xs font-normal text-gray-400">(click a crop to expand)</span>
          </p>
          {Object.entries(result.agronomic_diagnostics)
            .sort(([a], [b]) => (result.weights[b] ?? 0) - (result.weights[a] ?? 0))
            .map(([crop, reasons]) => {
              const d = result.crop_details[crop];
              const mult = d?.agronomic_multiplier ?? 1;
              const isExpanded = expandedCrop === crop;
              const suitabilityColor =
                mult >= 0.85 ? "bg-emerald-50 ring-emerald-200 text-emerald-700"
                : mult >= 0.60 ? "bg-blue-50 ring-blue-200 text-blue-700"
                : mult >= 0.35 ? "bg-amber-50 ring-amber-200 text-amber-700"
                : "bg-rose-50 ring-rose-200 text-rose-700";

              return (
                <div key={crop}
                  className={`rounded-xl ring-1 p-3 cursor-pointer transition-all ${suitabilityColor}`}
                  onClick={() => setExpandedCrop(isExpanded ? null : crop)}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-sm font-semibold">
                      <span>{mult >= 0.85 ? "✅" : mult >= 0.35 ? "⚠️" : "❌"}</span>
                      {crop}
                    </div>
                    <div className="flex items-center gap-3 text-xs">
                      <span className="font-bold">
                        Suitability: {(mult * 100).toFixed(0)}%
                      </span>
                      <span className="opacity-60">
                        Raw ML: {d?.raw_ml_yield?.toFixed(2)} → Adjusted: {d?.expected_yield?.toFixed(2)} t/ha
                      </span>
                      <span className="opacity-50">{isExpanded ? "▲" : "▼"}</span>
                    </div>
                  </div>

                  {/* Summary line (always visible — first reason) */}
                  {!isExpanded && reasons.length > 0 && (
                    <p className="text-xs opacity-75 mt-1 truncate">{reasons[0]}</p>
                  )}

                  {/* Expanded: all reasons */}
                  {isExpanded && (
                    <ul className="mt-2 space-y-1">
                      {reasons.map((r, i) => (
                        <li key={i} className="text-xs leading-relaxed">
                          {i === 0 ? <strong>{r}</strong> : <span>• {r}</span>}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              );
            })}
        </div>
      )}

      {/* Risk + Budget + Profit grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
        <div className="bg-gray-50 rounded-xl p-4 space-y-1">
          <p className="font-semibold text-gray-700">🌾 Dominant Crop</p>
          <p className="text-gray-600">
            Optimizer allocates <span className="font-bold">{dominantPct}%</span> ({dominantArea} ha)
            to <span className="font-bold text-emerald-700">{dominantCrop[0]}</span>
            {secondaryCrop && dominantCrop[1] < 0.99
              ? <>, with the rest toward <span className="font-semibold">{secondaryCrop[0]}</span>{sortedCrops.length > 2 ? " and others" : ""}.</>
              : " — single-crop strategy."}
          </p>
        </div>

        <div className="bg-gray-50 rounded-xl p-4 space-y-1">
          <p className="font-semibold text-gray-700">📊 Markowitz Risk & Diversification</p>
          <p className="text-gray-600">
            Portfolio strategy: <span className={`font-bold ${riskColor}`}>{riskLevel}</span>
          </p>
          <div className="text-xs text-gray-600 space-y-0.5 pt-1">
            <p>
              Portfolio Volatility: <span className="font-bold text-gray-900">{INR(result.portfolio_volatility_inr ?? 0)}</span>
            </p>
            <p>
              Sharpe Score: <span className="font-bold text-emerald-700">{result.sharpe_ratio ?? 0}</span>{" "}
              <span className="text-[11px] text-gray-400">(= [Net Profit − Budget] / Volatility)</span>
            </p>
            <p className="text-[10px] text-gray-400 font-mono">
              ({INR(netProfit)} − {INR(survivalBudget)}) / {INR(result.portfolio_volatility_inr || 1)}
            </p>
          </div>
          <p className="text-[11px] text-gray-400 pt-1">
            Quadratic risk penalty prevents single-crop failure using 28-yr {regimeBadge.label} climate covariance.
          </p>
        </div>

        <div className={`rounded-xl p-4 space-y-1 ${budgetMet ? "bg-gray-50" : "bg-rose-50"}`}>
          <p className="font-semibold text-gray-700">🛡️ Survival Budget</p>
          <p className="text-gray-600">
            {budgetMet
              ? <>Worst-case: <span className="font-bold text-emerald-700">{INR(result.worst_case_income)}</span> ≥ budget {INR(survivalBudget)} ✅</>
              : <>⚠ WC <span className="font-bold text-rose-700">{INR(result.worst_case_income)}</span> &lt; budget {INR(survivalBudget)}. Lower target or add higher-insured crops.</>
            }
          </p>
        </div>

        <div className="bg-gray-50 rounded-xl p-4 space-y-1">
          <p className="font-semibold text-gray-700">💰 Expected Net Profit</p>
          <p className="text-gray-600">
            After premiums {INR(result.total_premium)}, net income is{" "}
            <span className="font-bold text-emerald-700">{INR(netProfit)}</span>
            {" "}(≈ <span className="font-semibold">{INR(netProfit / totalAreaHa)}/ha</span>).
          </p>
        </div>
      </div>

      {/* PMFBY section */}
      {pmfbyInsights.length > 0 && (
        <div className="border-t border-gray-100 pt-4 space-y-2">
          <p className="text-sm font-semibold text-gray-700">🏦 PMFBY Insurance Payout Triggers</p>
          <p className="text-xs text-gray-500">Shortfall &gt; 80% of expected yield → payout fires in worst-case:</p>
          <ul className="space-y-1">
            {pmfbyInsights.map(({ crop, payout, premium }) => (
              <li key={crop} className="flex justify-between text-sm bg-blue-50 rounded-lg px-3 py-2">
                <span className="font-medium text-blue-800">{crop}</span>
                <span className="text-gray-600">
                  Payout <span className="font-semibold text-blue-700">{INR(payout)}</span>
                  &nbsp;|&nbsp;Premium <span className="font-semibold">{INR(premium)}</span>
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Agmarknet Price Risk & Natural Hedge Section */}
      {result.agmarknet_price_risk && Object.keys(result.agmarknet_price_risk).length > 0 && (
        <div className="border-t border-gray-100 pt-4 space-y-2">
          <div className="flex items-center justify-between">
            <p className="text-sm font-semibold text-gray-800 flex items-center gap-1.5">
              <span>📈</span> Agmarknet Mandi Price Risk & Natural Cobweb Hedge
            </p>
            <span className="text-[10px] text-gray-400 font-mono">Bohrnstedt-Goldberger Stochastic Layer</span>
          </div>
          <p className="text-xs text-gray-500 leading-relaxed">
            Revenue risk accounts for both yield volatility and mandi price swings. High negative correlation (<span className="font-mono text-emerald-700">ρ &lt; -0.20</span>) provides a natural hedge (bumper harvest lowers price; drought spikes price). Unhedged cash crops with global price exposure face amplified variance.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1">
            {Object.entries(result.agmarknet_price_risk)
              .filter(([c]) => (result.weights[c] ?? 0) > 0.01)
              .map(([crop, pr]) => (
                <div key={crop} className="bg-slate-50 border border-slate-200 rounded-xl p-3 text-xs space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-gray-900">{crop}</span>
                    <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                      pr.price_cv >= 0.25 ? "bg-rose-100 text-rose-700" :
                      pr.price_cv >= 0.15 ? "bg-amber-100 text-amber-700" :
                      "bg-emerald-100 text-emerald-700"
                    }`}>
                      Price CV: {(pr.price_cv * 100).toFixed(0)}%
                    </span>
                  </div>
                  <p className="text-gray-600 text-[11px]">
                    Policy: <span className="font-medium text-gray-800">{pr.policy}</span>
                  </p>
                  <p className="text-gray-600 text-[11px]">
                    Price-Yield Corr (<span className="font-mono">ρ</span>): <strong className={pr.rho_py < -0.15 ? "text-emerald-700" : "text-amber-700"}>{pr.rho_py > 0 ? `+${pr.rho_py}` : pr.rho_py}</strong> · {pr.natural_hedge}
                  </p>
                  <p className="text-[10px] text-gray-400">
                    Stochastic revenue variance: {pr.revenue_vol_multiplier >= 1.05 ? `+${((pr.revenue_vol_multiplier**2 - 1)*100).toFixed(0)}% risk inflation` : "stabilized by MSP floor"}
                  </p>
                </div>
              ))}
          </div>
        </div>
      )}

      <div className="border-t border-gray-100 pt-3">
        <p className="text-xs text-gray-400">
          ⚠ Academic decision-support tool (1997–2024 Indian crop data). Consult your local KVK before farming decisions.
        </p>
      </div>
    </div>
  );
}
