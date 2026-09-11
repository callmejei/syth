import { useEffect, useState } from 'react'
import { Link, NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import api from './api'
import {
  BoxIcon,
  DatabaseIcon,
  HomeIcon,
  ListIcon,
  SettingsIcon,
  ShieldIcon,
  SlidersIcon,
} from './components/icons'
import { ThemeToggle } from './components/theme'
import { LogoLockup } from './components/Logo'
import { useSession } from './components/session'
import Landing from './pages/Landing'
import Login from './pages/Login'
import Home from './pages/Home'
import DataSources from './pages/DataSources'
import Configurations from './pages/Configurations'
import ConfigDetail from './pages/ConfigDetail'
import Jobs from './pages/Jobs'
import Models from './pages/Models'
import ModelReport from './pages/ModelReport'
import Governance from './pages/Governance'
import Admin from './pages/Admin'
import ModelTypes from './pages/ModelTypes'
import ComputeSettings from './pages/ComputeSettings'
import AuditLogs from './pages/AuditLogs'
import ProjectDetail from './pages/ProjectDetail'

const NAV = [
  { to: '/app', label: 'Home', Icon: HomeIcon, end: true },
  { to: '/app/data-sources', label: 'Data sources', Icon: DatabaseIcon },
  { to: '/app/configurations', label: 'Configurations', Icon: SlidersIcon },
  { to: '/app/governance', label: 'Governance', Icon: ShieldIcon, badge: 'NEW' },
  { to: '/app/models', label: 'Models', Icon: BoxIcon },
  { to: '/app/jobs', label: 'My jobs', Icon: ListIcon },
  { to: '/app/admin', label: 'Admin settings', Icon: SettingsIcon },
]

function Sidebar() {
  const [projects, setProjects] = useState([])

  useEffect(() => {
    api.projects().then(setProjects).catch(() => setProjects([]))
  }, [])

  return (
    <aside className="flex h-screen w-64 shrink-0 flex-col border-r border-line bg-surface">
      <div className="px-5 py-5">
        <LogoLockup />
      </div>

      <nav aria-label="Main" className="flex-1 space-y-0.5 overflow-y-auto px-3">
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-md px-3 py-2 text-sm transition ${
                isActive
                  ? 'bg-accent-subtle font-medium text-accent-text'
                  : 'text-fg-muted hover:bg-sunken'
              }`
            }
          >
            <item.Icon size={18} className="shrink-0 text-fg-subtle" />
            <span className="flex-1">{item.label}</span>
            {item.badge && (
              <span className="pill bg-success-subtle text-success-text">{item.badge}</span>
            )}
          </NavLink>
        ))}

        <div className="pt-4">
          <div className="px-3 pb-1 text-[11px] font-semibold uppercase tracking-wide text-fg-subtle">
            Projects
          </div>
          {projects.map((project) => (
            <NavLink
              key={project.id}
              to={`/app/projects/${project.id}`}
              className={({ isActive }) =>
                `block rounded-md px-3 py-1.5 text-sm transition ${
                  isActive
                    ? 'bg-accent-subtle font-medium text-accent-text'
                    : 'text-fg-muted hover:bg-sunken hover:text-fg'
                }`
              }
            >
              {project.name}
            </NavLink>
          ))}
          {!projects.length && (
            <div className="px-3 py-1.5 text-sm text-fg-subtle">No projects</div>
          )}
        </div>
      </nav>

      <div className="space-y-3 border-t border-line px-5 py-4">
        <ThemeToggle />
        <div className="text-2xs leading-relaxed text-fg-subtle">
          Proof of concept · SPN and ARF engines
          <div>Group Data Office</div>
        </div>
      </div>
    </aside>
  )
}

function Breadcrumb() {
  const { pathname } = useLocation()
  // Strip the /app mount point: it is an implementation detail of the
  // route tree, not a place the user navigated to.
  const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
  const parts = pathname
    .split('/')
    .filter(Boolean)
    // Drop the /app mount point: it is an implementation detail of the route
    // tree, not a place the user navigated to.
    .filter((part, index) => !(index === 0 && part === 'app'))
    // Record ids carry no meaning in a trail. The page heading already names
    // the record, so a raw UUID is noise.
    .filter((part) => !UUID.test(part))
  if (!parts.length) return null
  return (
    <nav aria-label="Breadcrumb" className="mb-4 flex items-center gap-2 text-sm text-fg-muted">
      <Link to="/app" className="hover:text-accent-text">
        Home
      </Link>
      {parts.map((part, index) => (
        <span key={index} className="flex items-center gap-2">
          <span aria-hidden="true" className="text-fg-subtle">—</span>
          <span className="capitalize text-fg-muted">{part.replace(/-/g, ' ').slice(0, 24)}</span>
        </span>
      ))}
    </nav>
  )
}

function NotFound({ home = '/app' }) {
  // An unmatched route used to render nothing at all, which looks like a broken
  // build rather than a wrong address. Say what happened and offer a way out.
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center text-center">
      <div className="font-display text-5xl font-bold text-accent-text">404</div>
      <h1 className="mt-4 font-display text-xl font-semibold text-fg">
        This page does not exist
      </h1>
      <p className="mt-2 max-w-[46ch] text-sm text-fg-muted">
        The address may have changed, or the link that brought you here is out of
        date.
      </p>
      <Link to={home} className="btn-primary mt-6">
        Back to the workspace
      </Link>
    </div>
  )
}

function RequireSession({ children }) {
  const { user } = useSession()
  const location = useLocation()
  if (!user) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />
  }
  return children
}

function Workspace() {
  return (
    <div className="flex h-[100dvh] overflow-hidden">
      {/* Lets a keyboard user jump past the sidebar instead of tabbing through
          every nav item on every page. */}
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-[100] focus:rounded focus:bg-accent focus:px-3 focus:py-2 focus:text-sm focus:text-accent-fg"
      >
        Skip to main content
      </a>
      <Sidebar />
      <main id="main" tabIndex={-1} className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-[1400px] px-8 py-8">
          <Breadcrumb />
          <Routes>
            <Route index element={<Home />} />
            <Route path="data-sources" element={<DataSources />} />
            <Route path="configurations" element={<Configurations />} />
            <Route path="configurations/:id" element={<ConfigDetail />} />
            <Route path="governance" element={<Governance />} />
            <Route path="models" element={<Models />} />
            <Route path="models/:id" element={<ModelReport />} />
            <Route path="jobs" element={<Jobs />} />
            <Route path="admin" element={<Admin />} />
            <Route path="admin/model-types" element={<ModelTypes />} />
            <Route path="admin/compute" element={<ComputeSettings />} />
            <Route path="projects/:id" element={<ProjectDetail />} />
            <Route path="admin/audit-logs" element={<AuditLogs />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/login" element={<Login />} />
      <Route
        path="/app/*"
        element={
          <RequireSession>
            <Workspace />
          </RequireSession>
        }
      />
      <Route path="*" element={<NotFound home="/" />} />
    </Routes>
  )
}
