/**
 * MetricCard – a single KPI summary tile in the results dashboard.
 */
"use client";
import React from "react";

interface MetricCardProps {
  label: string;
  value: string;
  subLabel?: string;
  icon: React.ReactNode;
  color: "green" | "amber" | "blue" | "rose";
  warn?: boolean;
}

const COLOR_MAP = {
  green: { ring: "ring-emerald-200", bg: "bg-emerald-50",  iconBg: "bg-emerald-100", text: "text-emerald-700" },
  amber: { ring: "ring-amber-200",   bg: "bg-amber-50",    iconBg: "bg-amber-100",   text: "text-amber-700"  },
  blue:  { ring: "ring-blue-200",    bg: "bg-blue-50",     iconBg: "bg-blue-100",    text: "text-blue-700"   },
  rose:  { ring: "ring-rose-200",    bg: "bg-rose-50",     iconBg: "bg-rose-100",    text: "text-rose-700"   },
} as const;

export default function MetricCard({ label, value, subLabel, icon, color, warn = false }: MetricCardProps) {
  const c = COLOR_MAP[color];
  return (
    <div className={`relative rounded-2xl p-5 ring-1 shadow-sm transition-all duration-200 hover:shadow-md
      ${warn ? "ring-rose-400 bg-rose-50" : `${c.ring} ${c.bg}`}`}>
      {warn && (
        <span className="absolute top-3 right-3 text-xs font-semibold px-2 py-0.5 rounded-full bg-rose-200 text-rose-800">
          ⚠ Below Budget
        </span>
      )}
      <div className="flex items-start gap-4">
        <div className={`flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center text-xl
          ${warn ? "bg-rose-100" : c.iconBg}`}>
          {icon}
        </div>
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-1">{label}</p>
          <p className={`text-2xl font-bold truncate ${warn ? "text-rose-700" : c.text}`}>{value}</p>
          {subLabel && <p className="text-xs text-gray-500 mt-1 truncate">{subLabel}</p>}
        </div>
      </div>
    </div>
  );
}
