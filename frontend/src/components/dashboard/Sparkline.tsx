import { Line, LineChart } from "recharts";

export interface SparklineProps {
  data: number[];
  width?: number;
  height?: number;
  stroke?: string;
  strokeWidth?: number;
}

const defaultWidth = 120;
const defaultHeight = 32;
const defaultStroke = "rgb(34, 211, 238)";
const defaultStrokeWidth = 2;

export default function Sparkline({
  data,
  width = defaultWidth,
  height = defaultHeight,
  stroke = defaultStroke,
  strokeWidth = defaultStrokeWidth,
}: SparklineProps) {
  const resolvedWidth = width ?? defaultWidth;
  const resolvedHeight = height ?? defaultHeight;
  if (!data || data.length === 0) {
    return (
      <div
        className="flex items-center justify-center text-[11px] leading-[16px] text-[rgb(107,114,128)]"
        style={{ width: resolvedWidth, height: resolvedHeight }}
      >
        --
      </div>
    );
  }

  const chartData = data.map((value, index) => ({ index, value }));

  return (
    <LineChart
      width={resolvedWidth}
      height={resolvedHeight}
      data={chartData}
      margin={{ top: 0, right: 0, bottom: 0, left: 0 }}
    >
      <Line
        type="monotone"
        dataKey="value"
        stroke={stroke}
        strokeWidth={strokeWidth * 2.6}
        strokeOpacity={0.25}
        dot={false}
        isAnimationActive={false}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <Line
        type="monotone"
        dataKey="value"
        stroke={stroke}
        strokeWidth={strokeWidth}
        dot={false}
        isAnimationActive={false}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </LineChart>
  );
}
