import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import api from '../api'
import { PageHeader, Section, Spinner, Status, Table } from '../components/ui'

export default function Models() {
  const [rows, setRows] = useState(null)
  useEffect(() => {
    api.models().then(setRows)
  }, [])
  if (!rows) return <Spinner />

  return (
    <div>
      <PageHeader title="Models" subtitle="Trained SPN artifacts and their reports." />
      <Section title={`${rows.length} model(s)`}>
        <Table
          empty="No models trained yet."
          columns={[
            {
              key: 'name',
              label: 'Name',
              render: (row) => (
                <Link to={`/app/models/${row.id}`} className="text-accent-text hover:underline">
                  {row.name}
                </Link>
              ),
            },
            { key: 'configuration', label: 'Configuration' },
            { key: 'dataset', label: 'Dataset' },
            { key: 'model_type', label: 'Model type' },
            {
              key: 'fidelity',
              label: 'Fidelity',
              render: (row) =>
                row.fidelity != null ? (
                  <span
                    className={
                      row.fidelity > 0.85
                        ? 'text-success-text'
                        : row.fidelity > 0.7
                          ? 'text-warning-text'
                          : 'text-danger-text'
                    }
                  >
                    {(row.fidelity * 100).toFixed(1)}%
                  </span>
                ) : (
                  '—'
                ),
            },
            {
              key: 'dp_forced',
              label: 'DP',
              render: (row) =>
                row.dp_forced ? (
                  <span className="pill bg-violet-subtle text-violet-text">enforced</span>
                ) : (
                  <span className="text-xs text-fg-subtle">off</span>
                ),
            },
            { key: 'status', label: 'Status', render: (row) => <Status value={row.status} /> },
            {
              key: 'created_at',
              label: 'Created',
              render: (row) => new Date(row.created_at).toLocaleString(),
            },
          ]}
          rows={rows}
        />
      </Section>
    </div>
  )
}
