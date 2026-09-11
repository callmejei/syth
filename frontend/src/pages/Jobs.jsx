import { useEffect, useState } from 'react'
import api from '../api'
import { PageHeader, Section, Spinner, Status, Table } from '../components/ui'

export default function Jobs() {
  const [rows, setRows] = useState(null)

  useEffect(() => {
    function load() {
      api.jobs().then(setRows)
    }
    load()
    const timer = setInterval(load, 3000)
    return () => clearInterval(timer)
  }, [])

  if (!rows) return <Spinner />

  return (
    <div>
      <PageHeader title="My jobs" subtitle="Training and generation runs." />
      <Section title={`${rows.length} job(s)`}>
        <Table
          empty="No jobs yet."
          columns={[
            { key: 'job_type', label: 'Type' },
            { key: 'status', label: 'Status', render: (row) => <Status value={row.status} /> },
            {
              key: 'progress',
              label: 'Progress',
              render: (row) => (
                <div className="flex items-center gap-2">
                  <div className="h-1.5 w-24 overflow-hidden rounded bg-sunken">
                    <div
                      className="h-full bg-accent"
                      style={{ width: `${(row.progress || 0) * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-fg-subtle">
                    {Math.round((row.progress || 0) * 100)}%
                  </span>
                </div>
              ),
            },
            { key: 'message', label: 'Message' },
            { key: 'compute_profile', label: 'Profile' },
            {
              key: 'created_at',
              label: 'Started',
              render: (row) => new Date(row.created_at).toLocaleTimeString(),
            },
          ]}
          rows={rows}
        />
      </Section>
    </div>
  )
}
