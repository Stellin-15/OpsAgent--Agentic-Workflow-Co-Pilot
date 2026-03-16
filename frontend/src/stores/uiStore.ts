/**
 * Global UI state — Zustand store.
 *
 * Keeps transient UI state that doesn't belong in the server cache (TanStack
 * Query) or in URL params: sidebar state, selected incident, active filters.
 */

import { create } from 'zustand'
import type { IncidentStatus } from '../api/incidents'

interface UIState {
  // Sidebar
  sidebarOpen: boolean
  toggleSidebar: () => void

  // Incident selection
  selectedIncidentId: string | null
  selectIncident: (id: string | null) => void

  // Status filter (Incident Feed)
  statusFilter: IncidentStatus | ''
  setStatusFilter: (status: IncidentStatus | '') => void

  // Streaming state for the draft viewer
  isStreaming: boolean
  streamedContent: string
  streamError: string | null
  startStreaming: () => void
  appendToken: (token: string) => void
  finishStreaming: () => void
  setStreamError: (error: string) => void
  resetStream: () => void
}

export const useUIStore = create<UIState>((set) => ({
  sidebarOpen: true,
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),

  selectedIncidentId: null,
  selectIncident: (id) => set({ selectedIncidentId: id }),

  statusFilter: 'DRAFT_READY',
  setStatusFilter: (status) => set({ statusFilter: status }),

  isStreaming: false,
  streamedContent: '',
  streamError: null,

  startStreaming: () =>
    set({ isStreaming: true, streamedContent: '', streamError: null }),

  appendToken: (token) =>
    set((s) => ({ streamedContent: s.streamedContent + token })),

  finishStreaming: () => set({ isStreaming: false }),

  setStreamError: (error) =>
    set({ isStreaming: false, streamError: error }),

  resetStream: () =>
    set({ isStreaming: false, streamedContent: '', streamError: null }),
}))
