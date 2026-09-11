import { useCallback, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import api from '../api'
import { Banner, PageHeader, Section, Spinner, Status, Table } from '../components/ui'

function Empty({ children, action }) {
  return (
    <div className="rounded-lg border border-dashed border-line-strong px-6 py-8 text-center">
      <p className="text-sm text-fg-muted">{children}</p>
      {action && <div className="mt-3">{action}</div>}
    </div>
  )
}

export default function ProjectDetail() {
  const { id } = useParams()
  const [project, setProject] = useState(null)
  const [error, setError] = useState(null)

  const load = useCallback(() => {
    api
      .project(id)
      .then(setProject)
      .catch((err) => setError(err.message))
  }, [id])

  useEffect(load, [load])

  if (error) return <Banner tone="danger">{error}</Banner>
  if (!project) return <Spinner />

  const empty =
    !project.data_sources.length &&
    !project.configurations.length &&
    !project.models.length

  return (
    <div>
      <PageHeader
        title={project.name}
        subtitle={project.description || 'Project workspace'}
      />

      {empty && (
        <Banner tone="info" title="Nothing assigned to this project yet">
          Datasets and configurations are assigned to a project when they are
          created. Anything made before projects were wired up sits at the
          organisation level and will not appear here. Pick this project in the
          dialog when you upload a dataset or add a configuration.
        </Banner>
      )}

      <div className="grid gap-5 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <Section
            title="Models"
            description="Trained artifacts belonging to this project"
          >
            {project.models.length ? (
              <Table
                columns={[
                  {
                    key: 'name',
                    label: 'Name',
                    render: (row) => (
                      <Link
                        to={`/app/models/${row.id}`}
                        className="font-medium text-accent-text hover:underline"
                      >
                        {row.name}
                      </Link>
                    ),
                  },
                  { key: 'configuration', label: 'Configuration' },
                  { key: 'model_type', label: 'Engine' },
                  {
                    key: 'fidelity',
                    label: 'Fidelity',
                    numeric: true,
                    render: (row) =>
                      row.fidelity != null ? `${(row.fidelity * 100).toFixed(1)}%` : '—',
                  },
                  {
                    key: 'status',
                    label: 'Status',
                    render: (row) => <Status value={row.status} />,
                  },
                ]}
                rows={project.models}
              />
            ) : (
              <Empty
                action={
                  <Link to="/app/configurations" className="btn-ghost">
                    Go to configurations
                  </Link>
                }
              >
                No models trained in this project yet.
              </Empty>
            )}
          </Section>

          <Section title="Configurations">
            {project.configurations.length ? (
              <Table
                columns={[
                  {
                    key: 'name',
                    label: 'Name',
                    render: (row) => (
                      <Link
                        to={`/app/configurations/${row.id}`}
                        className="font-medium text-accent-text hover:underline"
                      >
                        {row.name}
                      </Link>
                    ),
                  },
                  { key: 'model_type', label: 'Engine' },
                  { key: 'tables', label: 'Tables', numeric: true },
                  {
                    key: 'status',
                    label: 'Status',
                    render: (row) => <Status value={row.status} />,
                  },
                ]}
                rows={project.configurations}
              />
            ) : (
              <Empty>No configurations in this project yet.</Empty>
            )}
          </Section>
        </div>

        <div>
          <Section title="Data sources">
            {project.data_sources.length ? (
              <ul className="space-y-3">
                {project.data_sources.map((source) => (
                  <li key={source.id} className="rounded-lg border border-line p-3">
                    <div className="font-medium text-fg">{source.name}</div>
                    <div className="num mt-0.5 text-xs text-fg-subtle">
                      {source.rows.toLocaleString()} rows across {source.tables.length}{' '}
                      table{source.tables.length === 1 ? '' : 's'}
                    </div>
                    <div className="mt-2 flex flex-wrap gap-1">
                      {source.tables.map((table) => (
                        <span key={table} className="pill bg-sunken text-fg-muted">
                          {table}
                        </span>
                      ))}
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <Empty
                action={
                  <Link to="/app/data-sources" className="btn-ghost">
                    Upload a dataset
                  </Link>
                }
              >
                No datasets assigned.
              </Empty>
            )}
          </Section>
        </div>
      </div>
    </div>
  )
}
