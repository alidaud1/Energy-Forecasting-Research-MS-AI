import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function ForecastChart(props: {
  title: string;
  data: any[];
  lines: { key: string; label: string; color: string; dashed?: boolean; width?: number }[];
}) {
  return (
    <div className="rounded-xl border border-neutral-200 overflow-hidden">
      <div className="px-3 py-2 bg-neutral-50 text-sm font-semibold">{props.title}</div>
      <div className="h-[260px] p-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={props.data}>
            <XAxis dataKey="t" hide />
            <YAxis width={48} />
            <Tooltip />
            {props.lines.map((l) => (
              <Line
                key={l.key}
                type="monotone"
                dataKey={l.key}
                name={l.label}
                stroke={l.color}
                dot={false}
                strokeWidth={l.width ?? 2}
                strokeDasharray={l.dashed ? "4 4" : undefined}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
