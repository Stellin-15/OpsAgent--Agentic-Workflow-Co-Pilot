/**
 * AnalyticsDashboard — MTTR trend, approval rate, and cost-per-incident charts.
 * Uses Recharts + TanStack Query.
 */

import { useQuery } from '@tanstack/react-query'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from 'recharts'
import { listIncidents } from '../../api/incidents'

const COLOURS = {
  approved: '#22c55e',
  rejected: '#f97316',
  critical: '#ef4444',
  high: '#f97316',
  warning: '#eab308',
  info: '#3b82f6',
}

export function AnalyticsDashboard() {
  const { data } = useQuery({
    queryKey: ['incidents', 'all'],
    queryFn: () => listIncidents(undefined, 100, 0),
    refetchInterval: 30_000,
  })

  const incidents = data?.items ?? []

  // ── Approval rate ───────────────────────────────────────────────────────
  const approved = incidents.filter((i) => i.status === 'RESOLVED').length
  const rejected = incidents.filter((i) => i.status === 'REJECTED').length
  const decisionPie = [
    { name: 'Approved', value: approved },
    { name: 'Rejected', value: rejected },
  ]

  // ── Incidents by severity ───────────────────────────────────────────────
  const bySeverity = ['critical', 'high', 'warning', 'info'].map((sev) => ({
    severity: sev,
    count: incidents.filter((i) => i.severity === sev).length,
  }))

  // ── Daily incident count (last 7 days) ──────────────────────────────────
  const now = Date.now()
  const dailyCounts = Array.from({ length: 7 }).map((_, i) => {
    const day = new Date(now - (6 - i) * 86400_000)
    const label = day.toLocaleDateString('en', { weekday: 'short' })
    const count = incidents.filter((inc) => {
      const d = new Date(inc.fired_at)
      return d.toDateString() === day.toDateString()
    }).length
    return { day: label, count }
  })

  // ── RAGAS confidence (from resolved incidents with drafts) ───────────────
  const confidenceData = incidents
    .flatMap((i) => i.drafts)
    .filter((d) => d.ragas_scores)
    .slice(0, 20)
    .map((d, idx) => ({
      idx,
      confidence: Math.round((d.ragas_scores!.confidence ?? 0) * 100),
    }))

  return (
    <div className="p-6 space-y-8">
      <h1 className="text-xl font-semibold text-gray-800">Analytics</h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard label="Total Incidents" value={incidents.length} />
        <StatCard label="Resolved" value={approved} colour="text-green-600" />
        <StatCard
          label="Approval Rate"
          value={
            approved + rejected > 0
              ? `${Math.round((approved / (approved + rejected)) * 100)}%`
              : '—'
          }
          colour="text-blue-600"
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Daily volume */}
        <ChartCard title="Incidents / Day (last 7 days)">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={dailyCounts}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="day" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Approval rate pie */}
        <ChartCard title="Draft Decisions">
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={decisionPie}
                cx="50%"
                cy="50%"
                innerRadius={50}
                outerRadius={80}
                dataKey="value"
                label={({ name, value }) => `${name}: ${value}`}
                labelLine={false}
              >
                <Cell fill={COLOURS.approved} />
                <Cell fill={COLOURS.rejected} />
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* By severity */}
        <ChartCard title="By Severity">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={bySeverity} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="severity" tick={{ fontSize: 11 }} width={60} />
              <Tooltip />
              <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                {bySeverity.map((entry) => (
                  <Cell
                    key={entry.severity}
                    fill={COLOURS[entry.severity as keyof typeof COLOURS] ?? '#6b7280'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* RAGAS confidence trend */}
        <ChartCard title="RAGAS Confidence (recent drafts)">
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={confidenceData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="idx" hide />
              <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} unit="%" />
              <Tooltip formatter={(v) => `${v}%`} />
              <Line
                type="monotone"
                dataKey="confidence"
                stroke="#8b5cf6"
                strokeWidth={2}
                dot={false}
              />
              {/* Confidence threshold line */}
              <Line
                type="monotone"
                dataKey={() => 60}
                stroke="#f97316"
                strokeDasharray="4 4"
                strokeWidth={1}
                dot={false}
                name="Threshold (60%)"
              />
              <Legend />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </div>
  )
}

function StatCard({
  label,
  value,
  colour = 'text-gray-800',
}: {
  label: string
  value: number | string
  colour?: string
}) {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <p className="text-xs text-gray-500 uppercase tracking-wide">{label}</p>
      <p className={`text-2xl font-bold mt-1 ${colour}`}>{value}</p>
    </div>
  )
}

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <h3 className="text-sm font-medium text-gray-700 mb-3">{title}</h3>
      {children}
    </div>
  )
}
