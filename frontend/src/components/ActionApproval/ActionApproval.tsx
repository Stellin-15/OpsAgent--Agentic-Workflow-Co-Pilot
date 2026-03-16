/**
 * ActionApproval — shows dry-run output and lets the user confirm or cancel
 * execution of a remediation action.
 */

import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../../api/client'
import { incidentKeys } from '../../api/incidents'

interface ActionOut {
  name: string
  description: string
  labels: string[]
  safety_level: string
  requires_approval: boolean
  has_dry_run: boolean
  parameters: Array<{ name: string; required: boolean; default: string | null; description: string }>
}

interface ExecutionResult {
  action_name: string
  command: string
  exit_code: number
  stdout: string
  stderr: string
  timed_out: boolean
  rolled_back: boolean
  succeeded: boolean
}

const SAFETY_COLOURS: Record<string, string> = {
  low: 'text-green-600 bg-green-50',
  medium: 'text-yellow-700 bg-yellow-50',
  high: 'text-orange-700 bg-orange-50',
  critical: 'text-red-700 bg-red-50',
}

export function ActionApproval({
  incidentId,
  action,
}: {
  incidentId: string
  action: ActionOut
}) {
  const [params, setParams] = useState<Record<string, string>>({})
  const [dryRunResult, setDryRunResult] = useState<ExecutionResult | null>(null)
  const [confirmed, setConfirmed] = useState(false)
  const queryClient = useQueryClient()

  const dryRunMutation = useMutation({
    mutationFn: () =>
      api.post<ExecutionResult>(`/actions/${action.name}/dry-run`, {
        incident_id: incidentId,
        params,
        dry_run: true,
        approved_by: 'ui-user',
      }),
    onSuccess: (result) => setDryRunResult(result),
  })

  const executeMutation = useMutation({
    mutationFn: () =>
      api.post<ExecutionResult>(`/actions/${action.name}/execute`, {
        incident_id: incidentId,
        params,
        dry_run: false,
        approved_by: 'ui-user',
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: incidentKeys.detail(incidentId) })
      setConfirmed(false)
      setDryRunResult(null)
    },
  })

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      {/* Action header */}
      <div className="flex items-center gap-3 px-4 py-3 bg-gray-50 border-b border-gray-200">
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-gray-900">{action.name}</span>
            <span
              className={`text-xs px-1.5 py-0.5 rounded font-medium ${
                SAFETY_COLOURS[action.safety_level] ?? 'text-gray-600 bg-gray-100'
              }`}
            >
              {action.safety_level}
            </span>
          </div>
          <p className="text-xs text-gray-500 mt-0.5">{action.description}</p>
        </div>
      </div>

      {/* Parameters */}
      {action.parameters.length > 0 && (
        <div className="px-4 py-3 border-b border-gray-100 space-y-2">
          {action.parameters.map((p) => (
            <div key={p.name} className="flex items-center gap-3">
              <label className="text-xs font-medium text-gray-600 w-32 shrink-0">
                {p.name}
                {p.required && <span className="text-red-400 ml-0.5">*</span>}
              </label>
              <input
                type="text"
                placeholder={p.default ?? p.description}
                value={params[p.name] ?? ''}
                onChange={(e) =>
                  setParams((prev) => ({ ...prev, [p.name]: e.target.value }))
                }
                className="flex-1 text-xs border border-gray-200 rounded px-2 py-1 font-mono"
              />
            </div>
          ))}
        </div>
      )}

      {/* Dry-run output */}
      {dryRunResult && (
        <div className="px-4 py-3 border-b border-gray-100">
          <p className="text-xs font-medium text-gray-600 mb-1">Dry-run output</p>
          <pre className="text-xs bg-gray-900 text-green-400 p-3 rounded overflow-x-auto whitespace-pre-wrap font-mono">
            $ {dryRunResult.command}{'\n'}
            {dryRunResult.stdout || dryRunResult.stderr || '(no output)'}
          </pre>
          {dryRunResult.exit_code !== 0 && (
            <p className="text-xs text-red-500 mt-1">
              Exit code: {dryRunResult.exit_code}
            </p>
          )}
        </div>
      )}

      {/* Action buttons */}
      <div className="flex items-center gap-2 px-4 py-3">
        {action.has_dry_run && !dryRunResult && (
          <button
            onClick={() => dryRunMutation.mutate()}
            disabled={dryRunMutation.isPending}
            className="text-xs px-3 py-1.5 border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50"
          >
            {dryRunMutation.isPending ? 'Running...' : 'Dry run'}
          </button>
        )}

        {(dryRunResult || !action.has_dry_run) && !confirmed && (
          <button
            onClick={() => setConfirmed(true)}
            className="text-xs px-3 py-1.5 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Execute
          </button>
        )}

        {confirmed && (
          <>
            <span className="text-xs text-orange-600 font-medium">
              Confirm: this will run for real
            </span>
            <button
              onClick={() => executeMutation.mutate()}
              disabled={executeMutation.isPending}
              className="text-xs px-3 py-1.5 bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
            >
              {executeMutation.isPending ? 'Running...' : 'Confirm execute'}
            </button>
            <button
              onClick={() => setConfirmed(false)}
              className="text-xs px-3 py-1.5 border border-gray-300 rounded hover:bg-gray-50"
            >
              Cancel
            </button>
          </>
        )}

        {executeMutation.isSuccess && (
          <span className="text-xs text-green-600 font-medium">Done</span>
        )}
        {executeMutation.isError && (
          <span className="text-xs text-red-600">
            Failed: {(executeMutation.error as Error).message}
          </span>
        )}
      </div>
    </div>
  )
}
