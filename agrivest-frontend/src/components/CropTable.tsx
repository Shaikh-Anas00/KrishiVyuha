/**
 * CropTable – per-crop financial breakdown table with allocation bars.
 */
"use client";
import React from "react";
import type { CropDetail } from "@/lib/api";

interface Props {
  cropDetails: Record<string, CropDetail>;
  weights    : Record<string, number>;
  areaHa     : Record<string, number>;
}

const INR = (n: number) =>
  new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(n);

export default function CropTable({ cropDetails, weights, areaHa }: Props) {
  const rows = Object.entries(cropDetails)
    .filter(([crop]) => (weights[crop] ?? 0) > 0.001)
    .sort(([a], [b]) => (weights[b] ?? 0) - (weights[a] ?? 0));

  return (
    <div className="rounded-2xl bg-white ring-1 ring-gray-200 shadow-sm overflow-hidden">
      <div className="px-6 py-4 border-b border-gray-100 flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-blue-100 flex items-center justify-center text-base">📋</div>
        <div>
          <h3 className="font-bold text-gray-900 text-sm">Per-Crop Financial Breakdown</h3>
          <p className="text-xs text-gray-500">Sorted by allocation weight · PMFBY payout shown where triggered</p>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-50 text-xs uppercase tracking-wide text-gray-500">
              <th className="px-5 py-3 text-left">Crop</th>
              <th className="px-4 py-3 text-right">Allocation</th>
              <th className="px-4 py-3 text-right">Area (ha)</th>
              <th className="px-4 py-3 text-right">E[Yield]</th>
              <th className="px-4 py-3 text-right">WC Yield</th>
              <th className="px-4 py-3 text-right">E[Revenue]</th>
              <th className="px-4 py-3 text-right">WC Net</th>
              <th className="px-4 py-3 text-right">PMFBY Payout</th>
              <th className="px-4 py-3 text-right">Premium</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {rows.map(([crop, d], idx) => {
              const wt = weights[crop] ?? 0;
              const ha = areaHa[crop] ?? 0;
              return (
                <tr key={crop} className={`transition-colors hover:bg-gray-50 ${idx === 0 ? "bg-emerald-50/40" : ""}`}>
                  <td className="px-5 py-3.5 font-semibold text-gray-800">
                    <div className="flex items-center gap-2">
                      {idx === 0 && <span className="text-xs bg-emerald-100 text-emerald-700 px-1.5 py-0.5 rounded-full">Top</span>}
                      {crop}
                    </div>
                  </td>
                  <td className="px-4 py-3.5 text-right">
                    <div className="flex items-center justify-end gap-2">
                      <div className="w-16 bg-gray-200 rounded-full h-1.5">
                        <div className="bg-emerald-500 h-1.5 rounded-full" style={{ width: `${wt * 100}%` }} />
                      </div>
                      <span className="font-semibold text-gray-700 w-12 text-right">{(wt * 100).toFixed(1)}%</span>
                    </div>
                  </td>
                  <td className="px-4 py-3.5 text-right text-gray-700">{ha.toFixed(2)}</td>
                  <td className="px-4 py-3.5 text-right text-gray-700">{d.expected_yield.toFixed(2)}</td>
                  <td className={`px-4 py-3.5 text-right font-medium ${d.worst_case_yield === 0 ? "text-rose-500" : "text-gray-700"}`}>
                    {d.worst_case_yield === 0 ? "0.00 ⚠" : d.worst_case_yield.toFixed(2)}
                  </td>
                  <td className="px-4 py-3.5 text-right text-emerald-700 font-medium">{INR(d.expected_revenue)}</td>
                  <td className={`px-4 py-3.5 text-right font-medium ${d.net_worst_case < 0 ? "text-rose-600" : "text-blue-600"}`}>
                    {INR(d.net_worst_case)}
                  </td>
                  <td className="px-4 py-3.5 text-right">
                    {d.pmfby_payout > 0 ? (
                      <span className="inline-flex items-center gap-1 text-blue-700 font-medium">
                        <span className="w-1.5 h-1.5 rounded-full bg-blue-500 inline-block" />
                        {INR(d.pmfby_payout)}
                      </span>
                    ) : <span className="text-gray-400">—</span>}
                  </td>
                  <td className="px-4 py-3.5 text-right text-gray-600">{INR(d.premium_cost)}</td>
                </tr>
              );
            })}
          </tbody>
          <tfoot className="bg-gray-50 border-t-2 border-gray-200 text-sm font-semibold text-gray-700">
            <tr>
              <td className="px-5 py-3" colSpan={5}>PORTFOLIO TOTAL</td>
              <td className="px-4 py-3 text-right text-emerald-700">{INR(rows.reduce((s, [, d]) => s + d.expected_revenue, 0))}</td>
              <td className="px-4 py-3 text-right text-blue-600">{INR(rows.reduce((s, [, d]) => s + d.net_worst_case, 0))}</td>
              <td className="px-4 py-3 text-right text-blue-600">{INR(rows.reduce((s, [, d]) => s + d.pmfby_payout, 0))}</td>
              <td className="px-4 py-3 text-right">{INR(rows.reduce((s, [, d]) => s + d.premium_cost, 0))}</td>
            </tr>
          </tfoot>
        </table>
      </div>
    </div>
  );
}
