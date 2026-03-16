/**
 * Incident API — typed wrappers over the FastAPI incident endpoints.
 */

import { api } from './client'

// ─── Types ────────────────────────────────────────────────────────────────────

export type IncidentStatus =
  | 'FIRING'
  | 'PROCESSING'
  | 'DRAFT_READY'
  | 'APPROVED'
  | 'EXECUTING'
  | 'RESOLVED'
  | 'REJECTED'
  | 'FAILED'

export type IncidentSeverity = 'critical' | 'high' | 'warning' | 'info' | 'unknown'

export interface Draft {
  id: string
  content: string
  model_used: string | null
  retrieval_score: number | null
  ragas_scores: RagasScores | null
  created_at: string
}

export interface RagasScores {
  context_precision: number
  faithfulness: number
  answer_relevance: number
  confidence: number
  low_confidence: boolean
}

export interface Incident {
  id: string
  alert_name: string
  status: IncidentStatus
  severity: IncidentSeverity
  source: string
  description: string | null
  labels: Record<string, string>
  fired_at: string
  resolved_at: string | null
  drafts: Draft[]
}

export interface IncidentListResponse {
  items: Incident[]
  total: number
}

// ─── Queries ──────────────────────────────────────────────────────────────────

export const incidentKeys = {
  all: ['incidents'] as const,
  list: (status?: string) => ['incidents', 'list', status] as const,
  detail: (id: string) => ['incidents', id] as const,
}

export async function listIncidents(
  statusFilter?: string,
  limit = 20,
  offset = 0,
): Promise<IncidentListResponse> {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) })
  if (statusFilter) params.set('status_filter', statusFilter)
  return api.get<IncidentListResponse>(`/incidents?${params}`)
}

export async function getIncident(id: string): Promise<Incident> {
  return api.get<Incident>(`/incidents/${id}`)
}

export async function approveIncident(id: string): Promise<Incident> {
  return api.post<Incident>(`/incidents/${id}/approve`)
}

export async function rejectIncident(
  id: string,
  reason: string,
  notes?: string,
): Promise<Incident> {
  return api.post<Incident>(`/incidents/${id}/reject`, { reason, notes })
}
