import { useEffect, useRef, useState } from 'react'
import api from '../api'
import { Banner, Modal, PageHeader, Section, Spinner, Table } from '../components/ui'
import { useDialog } from '../components/useDialog'
import { RefreshIcon, TrashIcon, UploadIcon } from '../components/icons'
import DownloadMenu from '../components/DownloadMenu'
import ConfirmDelete from '../components/ConfirmDelete'

function UploadModal({ onClose, onDone }) {
  const [name, setName] = useState('')
  const [projectId, setProjectId] = useState('')
  const [projects, setProjects] = useState([])
  const [files, setFiles] = useState([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const inputRef = useRef()

  useEffect(() => {
    api.projects().then(setProjects).catch(() => setProjects([]))
  }, [])

  async function submit() {
    if (!name || !files.length) return
    setBusy(true)
    setError(null)
    try {
      const result = await api.uploadDataSource(name, files, projectId)
      onDone(result)
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
        title="Add a new data source"
        description="Let's begin by defining your input data"
        onClose={onClose}
        footer={
          <>
            <button className="btn-ghost" onClick={onClose} type="button">
              Cancel
            </button>
            <button
              className="btn-primary"
              onClick={submit}
              disabled={busy || !name || !files.length}
              type="button"
            >
              {busy ? 'Uploading…' : 'Create data source'}
            </button>
          </>
        }
      >
        <div className="space-y-4">
          {error && <Banner tone="danger">{error}</Banner>}
          <div>
            <label className="label" htmlFor="ds-name">
              Data source name
            </label>
            <input
              id="ds-name"
              className="input mt-1"
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="retailbank_demo"
              aria-describedby="ds-name-help"
            />
            <p id="ds-name-help" className="mt-1 text-xs text-fg-muted">
              Used to identify this dataset in configurations.
            </p>
          </div>
          <div>
            <label className="label" htmlFor="ds-project">
              Project
            </label>
            <select
              id="ds-project"
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
          <div className="rounded-md border border-accent/30 bg-accent-subtle/50 p-4">
            <div className="text-sm font-medium text-fg-muted">
              I have data source files ready to upload
            </div>
            <button
              className="btn-ghost mt-3"
              type="button"
              onClick={() => inputRef.current?.click()}
            >
              <UploadIcon size={16} />
              Select one or multiple file(s)
            </button>
            <input
              ref={inputRef}
              id="ds-files"
              type="file"
              multiple
              accept=".csv,.parquet"
              className="sr-only"
              onChange={(event) => setFiles(Array.from(event.target.files || []))}
            />
            <div className="mt-3 text-xs text-fg-muted">
              Supported file formats:{' '}
              <span className="pill bg-sunken text-fg-muted">.csv</span>{' '}
              <span className="pill bg-sunken text-fg-muted">.parquet</span>
            </div>
            <div aria-live="polite">
              {files.length > 0 && (
                <ul className="mt-3 space-y-1 text-xs text-fg-muted">
                  {files.map((file) => (
                    <li key={file.name}>
                      {file.name} ({(file.size / 1024).toFixed(0)} KB)
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
          <p className="text-xs text-fg-muted">
            Each file becomes one table. Classifications are pulled from Apache Atlas on upload.
          </p>
        </div>
      </Modal>
    </div>
  )
}

export default function DataSources() {
  const [sources, setSources] = useState(null)
  const [modal, setModal] = useState(false)
  const [deleting, setDeleting] = useState(null)
  const [notice, setNotice] = useState(null)

  function load() {
    api.dataSources().then(setSources)
  }
  useEffect(load, [])

  if (!sources) return <Spinner />

  return (
    <div>
      <PageHeader
        title="Data sources"
        actions={
          <button className="btn-primary" onClick={() => setModal(true)} type="button">
            <UploadIcon size={16} />
            Add dataset
          </button>
        }
      />

      {notice && (
        <Banner tone="success" title={`Deleted "${notice.name}"`}>
          {notice.configurations || notice.models || notice.jobs
            ? `Also removed ${notice.configurations} configuration(s), ${notice.models} model(s) and ${notice.jobs} job(s).`
            : 'The dataset and its files are gone.'}
        </Banner>
      )}

      <Section title={`${sources.length} data source(s)`}>
        <Table
          empty="No data sources yet. Upload your related tables to begin."
          columns={[
            { key: 'name', label: 'Name' },
            {
              key: 'tables',
              label: 'Tables',
              render: (row) => (
                <div className="flex max-w-[200px] flex-wrap gap-1">
                  {row.tables.slice(0, 3).map((table) => (
                    <span
                      key={table}
                      className="pill max-w-[150px] bg-sunken text-fg-muted"
                      title={table}
                    >
                      {/* Table names are single unbreakable tokens, so the pill
                          itself has to clip or it sets the column width. */}
                      <span className="truncate">{table}</span>
                    </span>
                  ))}
                  {row.tables.length > 3 && (
                    <span
                      className="pill bg-sunken text-fg-subtle"
                      title={row.tables.join(', ')}
                    >
                      +{row.tables.length - 3} more
                    </span>
                  )}
                </div>
              ),
            },
            {
              key: 'row_count',
              label: 'Rows',
              render: (row) => row.row_count.toLocaleString(),
            },
            {
              key: 'atlas_source',
              label: 'Atlas',
              render: (row) => (
                <span
                  className={`pill ${
                    row.atlas_source === 'atlas'
                      ? 'bg-success-subtle text-success-text'
                      : 'bg-warning-subtle text-warning-text'
                  }`}
                >
                  {row.atlas_source || 'none'}
                </span>
              ),
            },
            {
              key: 'used_by',
              label: 'Used by',
              render: (row) =>
                row.used_by.length ? (
                  <span
                    className="block max-w-[150px] truncate"
                    title={row.used_by.join(', ')}
                  >
                    {row.used_by.join(', ')}
                  </span>
                ) : (
                  <span className="text-fg-subtle">N/A</span>
                ),
            },
            {
              key: 'actions',
              label: 'Actions',
              numeric: true,
              render: (row) => (
                <div className="flex items-center justify-end gap-0.5">
                  <DownloadMenu
                    label="Download"
                    title={
                      row.tables.length > 1
                        ? `Download all ${row.tables.length} tables as a zip`
                        : 'Download this table'
                    }
                    onDownload={(fmt) => api.downloadDataSource(row.id, { fmt })}
                  />
                  {/* Icon-only: three labelled buttons overflow the cell. The
                      label survives as the accessible name and the tooltip. */}
                  <button
                    className="btn-link px-1.5"
                    type="button"
                    title="Refresh Atlas classifications"
                    aria-label={`Refresh Atlas classifications for ${row.name}`}
                    onClick={() => api.refreshAtlas(row.id).then(load)}
                  >
                    <RefreshIcon size={14} />
                  </button>
                  <button
                    className="btn-link px-1.5 text-danger-text hover:bg-danger-subtle"
                    type="button"
                    onClick={() => setDeleting(row)}
                  >
                    <TrashIcon size={14} />
                    Delete
                  </button>
                </div>
              ),
            },
          ]}
          rows={sources}
        />
      </Section>

      {deleting && (
        <ConfirmDelete
          source={deleting}
          onClose={() => setDeleting(null)}
          onDeleted={(result) => {
            setDeleting(null)
            setNotice(result)
            load()
          }}
        />
      )}

      {modal && (
        <UploadModal
          onClose={() => setModal(false)}
          onDone={() => {
            setModal(false)
            load()
          }}
        />
      )}
    </div>
  )
}
