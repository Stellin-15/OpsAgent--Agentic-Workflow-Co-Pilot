/**
 * OpsAgent Dashboard — root component.
 *
 * Layout:
 *   Top nav  → links: Incidents | Analytics
 *   Content  → routed pages
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import { IncidentsPage } from './pages/IncidentsPage'
import { DashboardPage } from './pages/DashboardPage'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5_000,
      retry: 2,
    },
  },
})

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="flex flex-col h-screen bg-gray-50 text-gray-900">
          {/* Top navigation */}
          <header className="flex items-center gap-6 px-5 py-3 bg-white border-b border-gray-200 shadow-sm shrink-0">
            <div className="flex items-center gap-2">
              <span className="text-lg font-bold text-blue-600">OpsAgent</span>
              <span className="text-xs text-gray-400 font-medium">AI SRE Co-Pilot</span>
            </div>
            <nav className="flex gap-1 ml-4">
              <NavItem to="/" label="Incidents" />
              <NavItem to="/dashboard" label="Analytics" />
            </nav>
            <div className="ml-auto flex items-center gap-3">
              <StatusDot />
            </div>
          </header>

          {/* Page content */}
          <main className="flex-1 overflow-hidden">
            <Routes>
              <Route path="/" element={<IncidentsPage />} />
              <Route path="/dashboard" element={<DashboardPage />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  )
}

function NavItem({ to, label }: { to: string; label: string }) {
  return (
    <NavLink
      to={to}
      end
      className={({ isActive }) =>
        `text-sm px-3 py-1.5 rounded font-medium transition-colors ${
          isActive
            ? 'bg-blue-50 text-blue-700'
            : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
        }`
      }
    >
      {label}
    </NavLink>
  )
}

function StatusDot() {
  return (
    <div className="flex items-center gap-1.5 text-xs text-gray-500">
      <span className="w-2 h-2 rounded-full bg-green-400 inline-block" />
      Connected
    </div>
  )
}
