import { useEffect, useState } from 'react'
import api from '../api'
import { PageHeader, Section, Spinner, Table } from '../components/ui'

export default function AuditLogs() {
  const [rows, setRows] = useState(null)
  useEffect(() => {
    api.auditLogs().then(setRows)
  }, [])
  if (!rows) return <Spinner />

  return (
    <div>
      <PageHeader
        title="Audit Logs"
        subtitle="Traceability for every change that affects what data leaves the platform."
      />
      <Section title={`${rows.length} entries`}>
        <Table
          columns={[
            {
              key: 'created_at',
              label: 'When',
              render: (row) => new Date(row.created_at).toLocaleString(),
            },
            { key: 'actor', label: 'Actor' },
            { key: 'action', label: 'Action' },
            { key: 'entity_type', label: 'Entity' },
            {
              key: 'detail',
              label: 'Detail',
              render: (row) => (
                <code className="text-xs text-fg-subtle">
                  {JSON.stringify(row.detail).slice(0, 90)}
                </code>
              ),
            },
          ]}
          rows={rows}
        />
      </Section>
    </div>
  )
}
