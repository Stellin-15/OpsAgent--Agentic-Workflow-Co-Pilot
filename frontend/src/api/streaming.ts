/**
 * SSE streaming client.
 *
 * Opens an EventSource connection to /api/incidents/{id}/stream and
 * delivers tokens to a callback. Closes the connection on 'done' or 'error'.
 *
 * Usage:
 *   const stop = streamDraft(incidentId, (token) => setDraft(d => d + token))
 *   // later:
 *   stop()
 */

export type TokenCallback = (token: string) => void
export type DoneCallback = () => void
export type ErrorCallback = (error: string) => void

export function streamDraft(
  incidentId: string,
  onToken: TokenCallback,
  onDone?: DoneCallback,
  onError?: ErrorCallback,
): () => void {
  const url = `/api/incidents/${incidentId}/stream`
  const source = new EventSource(url)

  source.onmessage = (event) => {
    onToken(event.data)
  }

  source.addEventListener('done', () => {
    source.close()
    onDone?.()
  })

  source.addEventListener('error', (event) => {
    source.close()
    const messageEvent = event as MessageEvent
    const detail = messageEvent.data
      ? JSON.parse(messageEvent.data).detail
      : 'Stream error'
    onError?.(detail)
  })

  source.onerror = () => {
    source.close()
    onError?.('Connection lost')
  }

  // Return a stop function
  return () => source.close()
}
