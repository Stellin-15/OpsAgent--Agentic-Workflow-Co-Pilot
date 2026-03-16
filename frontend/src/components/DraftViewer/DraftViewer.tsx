/**
 * DraftViewer — streams LLM draft tokens in real-time with a RAGAS confidence
 * badge. Shows the typewriter effect as tokens arrive via SSE.
 *
 * Uses the Zustand stream state (appendToken, finishStreaming, etc.) so other
 * components can observe streaming state without prop drilling.
 */

import { useEffect, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import ReactMarkdown from 'react-markdown'
import { incidentKeys, getIncident, type RagasScores } from '../../api/incidents'
import { streamDraft } from '../../api/streaming'
import { useUIStore } from '../../stores/uiStore'

export function DraftViewer({ incidentId }: { incidentId: string }) {
  const {
    isStreaming,
    streamedContent,
    streamError,
    startStreaming,
    appendToken,
    finishStreaming,
    setStreamError,
    resetStream,
  } = useUIStore()

  const stopRef = useRef<(() => void) | null>(null)

  const { data: incident } = useQuery({
    queryKey: incidentKeys.detail(incidentId),
    queryFn: () => getIncident(incidentId),
  })

  const latestDraft = incident?.drafts?.[incident.drafts.length - 1]
  const ragas = latestDraft?.ragas_scores

  // Start streaming when incidentId changes
  useEffect(() => {
    resetStream()
    startStreaming()

    stopRef.current = streamDraft(
      incidentId,
      (token) => appendToken(token),
      () => finishStreaming(),
      (err) => setStreamError(err),
    )

    return () => {
      stopRef.current?.()
    }
  }, [incidentId])  // eslint-disable-line react-hooks/exhaustive-deps

  const content = streamedContent || latestDraft?.content || ''

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-gray-200">
        <h2 className="text-sm font-semibold text-gray-700">AI Draft</h2>
        {isStreaming && (
          <span className="flex items-center gap-1 text-xs text-blue-500">
            <span className="inline-block w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse" />
            Generating...
          </span>
        )}
        {ragas && <ConfidenceBadge scores={ragas} />}
        {latestDraft?.model_used && (
          <span className="ml-auto text-xs text-gray-400">
            {latestDraft.model_used}
          </span>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        {streamError ? (
          <div className="text-red-600 text-sm p-3 bg-red-50 rounded">
            Stream error: {streamError}
          </div>
        ) : content ? (
          <div className="prose prose-sm max-w-none">
            <ReactMarkdown>{content}</ReactMarkdown>
            {isStreaming && (
              <span className="inline-block w-1 h-4 bg-gray-400 animate-pulse ml-0.5" />
            )}
          </div>
        ) : (
          <div className="text-gray-400 text-sm text-center mt-8">
            Waiting for draft...
          </div>
        )}
      </div>
    </div>
  )
}

function ConfidenceBadge({ scores }: { scores: RagasScores }) {
  const pct = Math.round(scores.confidence * 100)
  const isLow = scores.low_confidence

  return (
    <div
      className={`flex items-center gap-1 text-xs px-2 py-0.5 rounded-full font-medium ${
        isLow
          ? 'bg-orange-100 text-orange-700'
          : 'bg-green-100 text-green-700'
      }`}
      title={`Context precision: ${(scores.context_precision * 100).toFixed(0)}%
Faithfulness: ${(scores.faithfulness * 100).toFixed(0)}%
Answer relevance: ${(scores.answer_relevance * 100).toFixed(0)}%`}
    >
      {isLow ? '⚠' : '✓'} {pct}% confidence
    </div>
  )
}
