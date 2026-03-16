/**
 * IncidentsPage — three-column layout:
 *   Left:   IncidentFeed (scrollable list)
 *   Center: DraftViewer (SSE streaming + markdown)
 *   Right:  Approve/Reject + action execution panel
 */

import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { IncidentFeed } from '../components/IncidentFeed/IncidentFeed'
import { DraftViewer } from '../components/DraftViewer/DraftViewer'
import { ActionApproval } from '../components/ActionApproval/ActionApproval'
import { useUIStore } from '../stores/uiStore'
import {
  incidentKeys,
  getIncident,
  approveIncident,
  rejectIncident,
} from '../api/incidents'
import { api } from '../api/client'

export function IncidentsPage() {
  const { selectedIncidentId } = useUIStore()

  return (
    <div className="flex h-full overflow-hidden">
      {/* Left — incident list */}
      <aside className="w-72 flex-shrink-0 border-r border-gray-200 overflow-hidden flex flex-col">
        <IncidentFeed />
      </aside>

      {/* Center — draft viewer */}
      <main className="flex-1 overflow-hidden flex flex-col">
        {selectedIncidentId ? (
          <DraftViewer incidentId={selectedIncidentId} />
        ) : (
          <EmptyState />
        )}
      </main>

      {/* Right — decision + actions panel */}
      {selectedIncidentId && (
        <aside className="w-80 flex-shrink-0 border-l border-gray-200 overflow-y-auto">
          <DecisionPanel incidentId={selectedIncidentId} />
        </aside>
      )}
    </div>
  )
}

function DecisionPanel({ incidentId }: { incidentId: string }) {
  const queryClient = useQueryClient()
  const [rejectReason, setRejectReason] = useState('')
  const [rejectNotes, setRejectNotes] = useState('')
  const [showRejectForm, setShowRejectForm] = useState(false)

  const { data: incident } = useQuery({
    queryKey: incidentKeys.detail(incidentId),
    queryFn: () => getIncident(incidentId),
  })

  const { data: actions } = useQuery({
    queryKey: ['actions'],
    queryFn: () => api.get<{ name: string; description: string; labels: string[]; safety_level: string; requires_approval: boolean; has_dry_run: boolean; parameters: unknown[] }[]>('/actions'),
  })

  const approveMutation = useMutation({
    mutationFn: () => approveIncident(incidentId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: incidentKeys.detail(incidentId) }),
  })

  const rejectMutation = useMutation({
    mutationFn: () => rejectIncident(incidentId, rejectReason, rejectNotes),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: incidentKeys.detail(incidentId) })
      setShowRejectForm(false)
    },
  })

  if (!incident) return null

  const isDraftReady = incident.status === 'DRAFT_READY'
  const isTerminal = ['RESOLVED', 'REJECTED', 'FAILED'].includes(incident.status)

  // Filter actions to only show kubernetes + aws labels
  const relevantActions = (actions ?? []).filter(
    (a) => a.labels.some((l) => ['kubernetes', 'aws', 'custom'].includes(l))
  ).slice(0, 3)

  return (
    <div className="p-4 space-y-4">
      {/* Status banner */}
      <div className="text-xs font-medium text-gray-500 uppercase tracking-wide">
        {incident.status.replace('_', ' ')}
      </div>

      {/* Approve / Reject */}
      {isDraftReady && !showRejectForm && (
        <div className="flex gap-2">
          <button
            onClick={() => approveMutation.mutate()}
            disabled={approveMutation.isPending}
            className="flex-1 py-2 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700 disabled:opacity-50 font-medium"
          >
            {approveMutation.isPending ? 'Approving...' : 'Approve'}
          </button>
          <button
            onClick={() => setShowRejectForm(true)}
            className="flex-1 py-2 border border-red-200 text-red-600 text-sm rounded-lg hover:bg-red-50 font-medium"
          >
            Reject
          </button>
        </div>
      )}

      {/* Reject form */}
      {showRejectForm && (
        <div className="space-y-2">
          <select
            value={rejectReason}
            onChange={(e) => setRejectReason(e.target.value)}
            className="w-full text-sm border border-gray-200 rounded px-2 py-1.5"
          >
            <option value="">Select reason...</option>
            <option value="hallucination">Hallucination</option>
            <option value="wrong_runbook">Wrong runbook</option>
            <option value="tone">Tone / style</option>
            <option value="incomplete">Incomplete</option>
            <option value="other">Other</option>
          </select>
          <textarea
            value={rejectNotes}
            onChange={(e) => setRejectNotes(e.target.value)}
            placeholder="Notes (optional)"
            className="w-full text-sm border border-gray-200 rounded px-2 py-1.5 h-20 resize-none"
          />
          <div className="flex gap-2">
            <button
              onClick={() => rejectMutation.mutate()}
              disabled={!rejectReason || rejectMutation.isPending}
              className="flex-1 py-1.5 bg-red-600 text-white text-sm rounded disabled:opacity-50"
            >
              Confirm Reject
            </button>
            <button
              onClick={() => setShowRejectForm(false)}
              className="flex-1 py-1.5 border border-gray-200 text-sm rounded"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {isTerminal && (
        <div className="text-sm text-gray-500 text-center py-2">
          Incident {incident.status.toLowerCase()}
        </div>
      )}

      {/* Action execution panel */}
      {relevantActions.length > 0 && (
        <div>
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
            Remediation Actions
          </h3>
          <div className="space-y-3">
            {relevantActions.map((action) => (
              <ActionApproval
                key={action.name}
                incidentId={incidentId}
                action={action as Parameters<typeof ActionApproval>[0]['action']}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-gray-400">
      <div className="text-4xl mb-3">🔍</div>
      <p className="text-sm">Select an incident to view the AI draft</p>
    </div>
  )
}
