import { useMemo } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { formatCurrency } from "../dashboard/formatters";
import type { EquityCurvePoint } from "./types";

interface EquityCurveCardProps {
  points: EquityCurvePoint[];
  isLoading?: boolean;
}

const axisColor = "var(--ds-text-secondary)";
const gridColor = "rgba(148, 163, 184, 0.22)";

export default function EquityCurveCard({ points, isLoading = false }: EquityCurveCardProps) {
  const chartData = useMemo(
    () =>
      points.map((point) => {
        const parsed = new Date(point.t);
        return {
          t: point.t,
          equity: point.equity,
          x:
            Number.isNaN(parsed.getTime())
              ? point.t
              : parsed.toLocaleDateString("en-US", { month: "short", day: "numeric" }),
        };
      }),
    [points]
  );

  return (
    <section className="rounded-2xl p-md shadow-card jbravo-panel jbravo-panel-cyan" aria-label="Equity curve">
      <header className="flex items-center justify-between gap-3">
        <div>
          <h2 className="font-arimo text-sm font-semibold uppercase tracking-[0.08em] text-primary">Equity Curve</h2>
          <p className="mt-1 text-xs text-secondary">Portfolio history from Alpaca paper account</p>
        </div>
      </header>

      <div className="mt-3 h-64">
        {isLoading ? (
          <div className="h-full animate-pulse rounded-xl jbravo-panel-inner jbravo-panel-inner-cyan" />
        ) : chartData.length === 0 ? (
          <div className="flex h-full items-center justify-center rounded-xl border border-dashed border-slate-300 text-sm text-secondary dark:border-slate-600">
            No portfolio history available.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
              <CartesianGrid stroke={gridColor} strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="x"
                tick={{ fill: axisColor, fontSize: 11 }}
                axisLine={{ stroke: gridColor }}
                tickLine={{ stroke: gridColor }}
                minTickGap={24}
              />
              <YAxis
                width={80}
                tick={{ fill: axisColor, fontSize: 11 }}
                axisLine={{ stroke: gridColor }}
                tickLine={{ stroke: gridColor }}
                tickFormatter={(value) => formatCurrency(Number(value))}
              />
              <Tooltip
                cursor={{ stroke: "rgba(56, 189, 248, 0.45)", strokeWidth: 1 }}
                contentStyle={{
                  borderRadius: "0.5rem",
                  borderColor: "var(--ds-outline-subtle)",
                  backgroundColor: "var(--ds-surface)",
                }}
                labelStyle={{ color: "var(--ds-text-secondary)", fontSize: "0.75rem" }}
                formatter={(value: number) => formatCurrency(value)}
              />
              <Line
                type="monotone"
                dataKey="equity"
                stroke="#38bdf8"
                strokeWidth={2.25}
                dot={false}
                activeDot={{ r: 4, strokeWidth: 0, fill: "#38bdf8" }}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </section>
  );
}
