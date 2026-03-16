/**
 * Playwright e2e test — full incident workflow.
 *
 * Prerequisites:
 *   - OpsAgent API running at localhost:8000
 *   - Vite dev server running at localhost:5173
 *   - Test database seeded (docker compose up postgres redis -d + alembic upgrade head)
 *
 * Run: npm run test:e2e
 */

import { test, expect, type Page } from '@playwright/test'

const API_BASE = 'http://localhost:8000'

async function createTestIncident(page: Page) {
  const response = await page.request.post(`${API_BASE}/api/alerts/webhook/manual`, {
    data: {
      alert_name: 'E2ETestHighCPU',
      severity: 'critical',
      description: 'CPU at 98% on web-01',
      labels: { instance: 'web-01', env: 'e2e' },
    },
  })
  expect(response.status()).toBe(202)
  return response.json()
}

test.describe('Incident workflow', () => {
  test('landing page shows incident feed', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByText('OpsAgent')).toBeVisible()
    await expect(page.getByText('Incidents')).toBeVisible()
  })

  test('fire alert → incident appears in feed', async ({ page }) => {
    await createTestIncident(page)

    await page.goto('/')
    // Wait for the feed to reload (refetchInterval = 10s, but reload faster)
    await page.waitForTimeout(1000)
    await page.reload()

    await expect(page.getByText('E2ETestHighCPU')).toBeVisible({ timeout: 15_000 })
  })

  test('select incident → draft viewer opens', async ({ page }) => {
    await createTestIncident(page)
    await page.goto('/')
    await page.waitForTimeout(1000)
    await page.reload()

    const incidentRow = page.getByText('E2ETestHighCPU').first()
    await incidentRow.waitFor({ timeout: 15_000 })
    await incidentRow.click()

    // Draft viewer should appear
    await expect(page.getByText('AI Draft')).toBeVisible({ timeout: 5_000 })
  })

  test('reject incident with reason', async ({ page }) => {
    const incident = await createTestIncident(page)
    const incidentId: string = incident.id

    // Wait for DRAFT_READY state
    await page.waitForTimeout(2000)

    await page.goto('/')
    await page.reload()

    const incidentRow = page.getByText('E2ETestHighCPU').first()
    await incidentRow.waitFor({ timeout: 15_000 })
    await incidentRow.click()

    // Click reject
    await page.getByRole('button', { name: 'Reject' }).click()

    // Select reason
    await page.locator('select').selectOption('hallucination')

    // Confirm
    await page.getByRole('button', { name: 'Confirm Reject' }).click()

    // Incident should show REJECTED
    await expect(page.getByText('REJECTED', { exact: false })).toBeVisible({ timeout: 10_000 })

    // Verify via API
    const detail = await page.request.get(`${API_BASE}/api/incidents/${incidentId}`)
    const data = await detail.json()
    expect(data.status).toBe('REJECTED')
  })

  test('analytics page loads with charts', async ({ page }) => {
    await page.goto('/dashboard')
    await expect(page.getByText('Analytics')).toBeVisible()
    await expect(page.getByText('Total Incidents')).toBeVisible()
  })
})
