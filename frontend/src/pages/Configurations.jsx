import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import api from '../api'
import { Banner, Modal, PageHeader, Section, Spinner, Status, Table } from '../components/ui'
import { useDialog } from '../components/useDialog'
import { PlusIcon } from '../components/icons'

function AddModal({ onClose, onDone }) {
  const [name, setName] = useState('')
  const [modelType, setModelType] = useState('SPN')
  const [types, setTypes] = useState([])
  const [sources, setSources] = useState([])
  const [sourceId, setSourceId] = useState('')
  const [tables, setTables] = useState([])
  const [selected, setSelected] = useState([])
  const [projectId, setProjectId] = useState('')
  const [projects, setProjects] = useState([])
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    api.modelTypes().then(setTypes)
    api.dataSources().then(setSources)
    api.projects().then(setProjects).catch(() => setProjects([]))
  }, [])

  useEffect(() => {
    if (!sourceId) return setTables([])
    api.dataSource(sourceId).then((source) => {
      const names = source.tables.map((table) => table.name)
      setTables(names)
      setSelected(names)
    })
  }, [sourceId])

  const activeType = types.find((type) => type.name === modelType)

  async function submit() {
    setBusy(true)
    setError(null)
    try {
      const created = await api.createConfiguration({
        name,
        model_type: modelType,
        data_source_id: sourceId || null,
        project_id: projectId || null,
        selected_tables: selected,
      })
      onDone(created)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const dialogRef = useDialog(onClose)

  return (
    <div ref={dialogRef}>
      <Modal
        title="Add configuration"
        description="Let's add a configuration to start training"
        onClose={onClose}
        footer={
          <>
            <button className="btn-ghost" onClick={onClose} type="button">
              Cancel
            </button>
            <button
              className="btn-primary"
              onClick={submit}
              disabled={busy || !name || !activeType?.enabled}
              type="button"
            >
              Create configuration
            </button>
          </>
        }
      >
        <div className="space-y-4">
          {error && <Banner tone="danger">{error}</Banner>}
          <div>
            <label className="label" htmlFor="cfg-name">
              Configuration name
            </label>
            <input
              id="cfg-name"
              className="input mt-1"
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="PassingFuchsiaPanda"
            />
          </div>
          <div>
            <label className="label" htmlFor="cfg-model">
              Model type
            </label>
            <select
              id="cfg-model"
              className="input mt-1"
              value={modelType}
              onChange={(event) => setModelType(event.target.value)}
              aria-describedby="cfg-model-help"
            >
              {types.map((type) => (
                <option key={type.id} value={type.name} disabled={!type.enabled}>
                  {type.name} {type.enabled ? '' : '(disabled)'}
                </option>
              ))}
            </select>
            {activeType && (
              <p id="cfg-model-help" className="mt-1.5 text-xs text-fg-muted">
                {activeType.description}
              </p>
            )}
            {activeType && !activeType.enabled && (
              <Banner tone="warn">{activeType.licence_note}</Banner>
            )}
          </div>
          <div>
            <label className="label" htmlFor="cfg-project">
              Project
            </label>
            <select
              id="cfg-project"
              className="input mt-1"
              value={projectId}
              onChange={(event) => setProjectId(event.target.value)}
            >
              <option value="">No project (organisation level)</option>
              {projects.map((project) => (
                <option key={project.id} value={project.id}>
                  {project.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="label" htmlFor="cfg-dataset">
              Dataset
            </label>
            <select
              id="cfg-dataset"
              className="input mt-1"
              value={sourceId}
              onChange={(event) => setSourceId(event.target.value)}
            >
              <option value="">- select -</option>
              {sources.map((source) => (
                <option key={source.id} value={source.id}>
                  {source.name}
                </option>
              ))}
            </select>
            {tables.length > 0 && (
              <fieldset className="mt-3">
                <legend className="text-xs font-medium text-fg-muted">Tables to include</legend>
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  {tables.map((table) => {
                    const on = selected.includes(table)
                    return (
                      <button
                        key={table}
                        type="button"
                        aria-pressed={on}
                        onClick={() =>
                          setSelected((current) =>
                            current.includes(table)
                              ? current.filter((t) => t !== table)
                              : [...current, table],
                          )
                        }
                        className={`inline-flex min-h-[32px] cursor-pointer items-center rounded-full border px-3 text-xs font-medium ${
                          on
                            ? 'border-accent/40 bg-accent-subtle text-accent-text'
                            : 'border-line-strong text-fg-muted hover:bg-sunken'
                        }`}
                      >
                        {table}
                      </button>
                    )
                  })}
                </div>
              </fieldset>
            )}
          </div>
        </div>
      </Modal>
    </div>
  )
}

export default function Configurations() {
  const [rows, setRows] = useState(null)
  const [modal, setModal] = useState(false)
  const navigate = useNavigate()

  function load() {
    api.configurations().then(setRows)
  }
  useEffect(load, [])

  if (!rows) return <Spinner />

  return (
    <div>
      <PageHeader
        title="Configurations"
        actions={
          <button className="btn-primary" onClick={() => setModal(true)} type="button">
            <PlusIcon size={16} />
            Add Configuration
          </button>
        }
      />
      <Section title={`${rows.length} configuration(s)`}>
        <Table
          empty="No configurations yet."
          columns={[
            {
              key: 'name',
              label: 'Name',
              render: (row) => (
                <Link to={`/app/configurations/${row.id}`} className="text-accent-text hover:underline">
                  {row.name}
                </Link>
              ),
            },
            { key: 'model_type', label: 'Model' },
            {
              key: 'selected_tables',
              label: 'Tables',
              render: (row) => row.selected_tables?.length || 0,
            },
            { key: 'status', label: 'Status', render: (row) => <Status value={row.status} /> },
            {
              key: 'updated_at',
              label: 'Last edited',
              render: (row) =>
                row.updated_at ? new Date(row.updated_at).toLocaleString() : '—',
            },
          ]}
          rows={rows}
        />
      </Section>

      {modal && (
        <AddModal
          onClose={() => setModal(false)}
          onDone={(created) => navigate(`/app/configurations/${created.id}`)}
        />
      )}
    </div>
  )
}
