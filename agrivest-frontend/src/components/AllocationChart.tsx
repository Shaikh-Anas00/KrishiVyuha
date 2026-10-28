/**
 * AllocationChart – Recharts PieChart (donut) showing crop weight percentages.
 *
 * Install recharts first:  npm install recharts
 */
"use client";
import React, { useState } from "react";
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer, Sector } from "recharts";

const PALETTE = [
  "#10b981","#3b82f6","#f59e0b","#8b5cf6","#ef4444",
  "#06b6d4","#f97316","#84cc16","#ec4899","#6366f1",
];

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function ActiveShape(props: any) {
  const { cx, cy, innerRadius, outerRadius, startAngle, endAngle, fill, payload, percent } = props;
  return (
    <g>
      <text x={cx} y={cy - 10} textAnchor="middle" fill={fill} fontSize={13} fontWeight={700}>
        {payload.name}
      </text>
      <text x={cx} y={cy + 12} textAnchor="middle" fill="#6b7280" fontSize={12}>
        {(percent * 100).toFixed(1)}%
      </text>
      <Sector cx={cx} cy={cy} innerRadius={innerRadius} outerRadius={outerRadius + 10}
        startAngle={startAngle} endAngle={endAngle} fill={fill} />
      <Sector cx={cx} cy={cy} innerRadius={outerRadius + 14} outerRadius={outerRadius + 18}
        startAngle={startAngle} endAngle={endAngle} fill={fill} />
    </g>
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const { name, value, payload: inner } = payload[0];
  return (
    <div className="bg-white border border-gray-200 rounded-xl px-4 py-3 shadow-lg text-sm">
      <p className="font-bold text-gray-800">{name}</p>
      <p className="text-gray-600">Allocation: <span className="font-semibold">{(value * 100).toFixed(2)}%</span></p>
      <p className="text-gray-600">Area: <span className="font-semibold">{inner.area.toFixed(2)} ha</span></p>
    </div>
  );
}

interface AllocationChartProps {
  weights: Record<string, number>;
  areaHa : Record<string, number>;
}

export default function AllocationChart({ weights, areaHa }: AllocationChartProps) {
  const [activeIndex, setActiveIndex] = useState<number>(0);

  const data = Object.entries(weights)
    .filter(([, w]) => w > 0.001)
    .sort(([, a], [, b]) => b - a)
    .map(([crop, weight]) => ({ name: crop, value: weight, area: areaHa[crop] ?? 0 }));

  if (!data.length) return (
    <div className="flex items-center justify-center h-64 text-gray-400 text-sm">
      No allocation data to display.
    </div>
  );

  return (
    <div className="w-full h-72">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={68}
            outerRadius={100}
            dataKey="value"
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            activeShape={(props: any) => {
              const idx = data.findIndex((d) => d.name === props.name);
              return idx === activeIndex ? <ActiveShape {...props} /> : <Sector {...props} />;
            }}
            onMouseEnter={(_, index) => setActiveIndex(index)}
          >
            {data.map((_, idx) => (
              <Cell key={idx} fill={PALETTE[idx % PALETTE.length]} stroke="white" strokeWidth={2} />
            ))}
          </Pie>
          <Tooltip content={<CustomTooltip />} />
          <Legend iconType="circle" iconSize={10}
            formatter={(v: string) => <span className="text-xs text-gray-700">{v}</span>} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
