/**
 * IncidentFeed — real-time list of active incidents.
 *
 * Polls every 10 seconds via TanStack Query (refetchInterval).
 * Selecting an incident updates the Zustand store and triggers the
 * DraftViewer to load that incident's draft.
 */

import { useQuery } from '@tanstack/react-query'
import { formatDistanceToNow } from 'date-fns'
import { incidentKeys, listIncidents, type Incident, type IncidentStatus } from '../../api/incidents'
import { useUIStore } from '../../stores/uiStore'

const STATUS_COLOURS: Record<IncidentStatus, string> = {
  FIRING: 'bg-red-100 text-red-800',
  PROCESSING: 'bg-yellow-100 text-yellow-800',
  DRAFT_READY: 'bg-blue-100 text-blue-800',
  APPROVED: 'bg-green-100 text-green-800',
  EXECUTING: 'bg-purple-100 text-purple-800',
  RESOLVED: 'bg-gray-100 text-gray-600',
  REJECTED: 'bg-orange-100 text-orange-800',
  FAILED: 'bg-red-200 text-red-900',
}

const SEVERITY_DOT: Record<string, string> = {
  critical: 'bg-red-500',
  high: 'bg-orange-400',
  warning: 'bg-yellow-400',
  info: 'bg-blue-400',
  unknown: 'bg-gray-400',
}

export function IncidentFeed() {
  const { statusFilter, selectedIncidentId, selectIncident } = useUIStore()

  const { data, isLoading, error } = useQuery({
    queryKey: incidentKeys.list(statusFilter || undefined),
    queryFn: () => listIncidents(statusFilter || undefined),
    refetchInterval: 10_000,
  })

  if (isLoading) return <FeedSkeleton />
  if (error) return <div className="p-4 text-red-600">Failed to load incidents</div>

  const incidents = data?.items ?? []

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
        <h2 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">
          Incidents
          {data && <span className="ml-2 text-gray-400">({data.total})</span>}
        </h2>
        <StatusFilterPicker />
      </div>

      <div className="flex-1 overflow-y-auto">
        {incidents.length === 0 ? (
          <div className="p-6 text-center text-gray-400 text-sm">
            No incidents matching filter
          </div>
        ) : (
          incidents.map((inc) => (
            <IncidentRow
              key={inc.id}
              incident={inc}
              selected={inc.id === selectedIncidentId}
              onSelect={() => selectIncident(inc.id)}
            />
          ))
        )}
      </div>
    </div>
  )
}

function IncidentRow({
  incident,
  selected,
  onSelect,
}: {
  incident: Incident
  selected: boolean
  onSelect: () => void
}) {
  return (
    <button
      onClick={onSelect}
      className={`w-full text-left px-4 py-3 border-b border-gray-100 hover:bg-gray-50 transition-colors ${
        selected ? 'bg-blue-50 border-l-2 border-l-blue-500' : ''
      }`}
    >
      <div className="flex items-center gap-2 mb-1">
        <span
          className={`inline-block w-2 h-2 rounded-full flex-shrink-0 ${
            SEVERITY_DOT[incident.severity] ?? 'bg-gray-400'
          }`}
        />
        <span className="text-sm font-medium text-gray-900 truncate">
          {incident.alert_name}
        </span>
        <span
          className={`ml-auto text-xs font-medium px-1.5 py-0.5 rounded ${
            STATUS_COLOURS[incident.status]
          }`}
        >
          {incident.status.replace('_', ' ')}
        </span>
      </div>
      <div className="flex items-center gap-2 text-xs text-gray-400">
        <span>{incident.source}</span>
        <span>·</span>
        <span>
          {formatDistanceToNow(new Date(incident.fired_at), { addSuffix: true })}
        </span>
      </div>
    </button>
  )
}

function StatusFilterPicker() {
  const { statusFilter, setStatusFilter } = useUIStore()
  const options: Array<{ value: IncidentStatus | ''; label: string }> = [
    { value: '', label: 'All' },
    { value: 'FIRING', label: 'Firing' },
    { value: 'DRAFT_READY', label: 'Draft ready' },
    { value: 'RESOLVED', label: 'Resolved' },
  ]

  return (
    <select
      value={statusFilter}
      onChange={(e) => setStatusFilter(e.target.value as IncidentStatus | '')}
      className="text-xs border border-gray-200 rounded px-2 py-1 bg-white text-gray-600"
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  )
}

function FeedSkeleton() {
  return (
    <div className="p-4 space-y-3">
      {[1, 2, 3].map((i) => (
        <div key={i} className="animate-pulse space-y-2">
          <div className="h-4 bg-gray-200 rounded w-3/4" />
          <div className="h-3 bg-gray-100 rounded w-1/2" />
        </div>
      ))}
    </div>
  )
}
