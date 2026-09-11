import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import api from '../api'
import { Metric, PageHeader } from '../components/ui'
import { ShieldIcon, SlidersIcon, UploadIcon } from '../components/icons'

const QUICK_ACTIONS = [
  {
    to: '/app/data-sources',
    title: 'Upload Dataset',
    body: 'Load related tables and pull their Apache Atlas classifications.',
    Icon: UploadIcon,
  },
  {
    to: '/app/configurations',
    title: 'Create Configuration',
    body: 'Set the relational schema and SPN parameters for a training run.',
    Icon: SlidersIcon,
  },
  {
    to: '/app/governance',
    title: 'Review Data Policy',
    body: 'See what Atlas says is sensitive and how each column will be treated.',
    Icon: ShieldIcon,
  },
]

export default function Home() {
  const [stats, setStats] = useState({})
  const [atlas, setAtlas] = useState(null)

  useEffect(() => {
    Promise.all([
      api.dataSources().catch(() => []),
      api.configurations().catch(() => []),
      api.models().catch(() => []),
      api.jobs().catch(() => []),
    ]).then(([sources, configurations, models, jobs]) =>
      setStats({ sources, configurations, models, jobs }),
    )
    api.atlasStatus().then(setAtlas).catch(() => setAtlas({ reachable: false }))
  }, [])

  return (
    <div>
      <PageHeader
        title="Welcome to OCBC DataCraft"
        subtitle="Generate synthetic data with privacy, precision and scale — governed by your own data catalog."
      />

      <div className="mb-6 flex flex-wrap gap-2">
        <span className="pill bg-accent-subtle text-accent-text">SPN ENGINE</span>
        <span className="pill bg-success-subtle text-success-text">ATLAS GOVERNED</span>
        <span className="pill bg-sunken text-fg-muted">RELATIONAL</span>
        <span className="pill bg-warning-subtle text-warning-text">POC</span>
      </div>

      <div className="mb-6 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Metric label="Data sources" value={stats.sources?.length ?? '—'} />
        <Metric label="Configurations" value={stats.configurations?.length ?? '—'} />
        <Metric label="Trained models" value={stats.models?.length ?? '—'} />
        <Metric
          label="Apache Atlas"
          value={atlas?.reachable ? 'Live' : 'Fixture'}
          tone={atlas?.reachable ? 'text-success-text' : 'text-warning-text'}
          hint={atlas?.reachable ? atlas.base_url : 'falling back to bundled fixture'}
        />
      </div>

      <h2 className="mb-3 text-lg font-semibold text-fg">Quick Actions</h2>
      <p className="mb-4 text-sm text-fg-muted">to start generating synthetic data</p>

      <div className="grid gap-4 md:grid-cols-3">
        {QUICK_ACTIONS.map((action) => (
          <Link
            key={action.to}
            to={action.to}
            className="card group p-5 transition hover:border-accent/40 hover:shadow"
          >
            <div className="mb-3 flex h-9 w-9 items-center justify-center rounded-md bg-accent-subtle text-accent-text">
              <action.Icon size={18} />
            </div>
            <div className="font-medium text-fg group-hover:text-accent-text">
              {action.title}
            </div>
            <p className="mt-1 text-sm text-fg-muted">{action.body}</p>
          </Link>
        ))}
      </div>
    </div>
  )
}
