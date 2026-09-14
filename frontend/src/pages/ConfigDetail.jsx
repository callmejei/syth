import { useCallback, useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import api from '../api'
import FieldTree from '../components/FieldTree'
import { Banner, Section, Spinner, Status, Toggle } from '../components/ui'
import { PlayIcon, ShieldIcon } from '../components/icons'

export default function ConfigDetail() {
  const { id } = useParams()
  const navigate = useNavigate()

  const [config, setConfig] = useState(null)
  const [source, setSource] = useState(null)
  const [tree, setTree] = useState(null)
  const [dataArgs, setDataArgs] = useState({})
  const [training, setTraining] = useState({})
  const [filter, setFilter] = useState('high')
  const [jsonView, setJsonView] = useState(false)
  const [message, setMessage] = useState(null)
  const [busy, setBusy] = useState(false)
  const [job, setJob] = useState(null)
  // Set when Auto-configure refused because the catalog does not cover these
  // tables, so the fallback can be offered instead of a dead end.
  const [catalogGap, setCatalogGap] = useState(false)

  const load = useCallback(async () => {
    const configuration = await api.configuration(id)
    setConfig(configuration)
    setDataArgs(configuration.data_args || {})
    setTraining(configuration.training_params || {})
    if (configuration.data_source_id) {
      const [dataSource, schema] = await Promise.all([
        api.dataSource(configuration.data_source_id),
        api.schemaTree(configuration.data_source_id, configuration.selected_tables),
      ])
      setSource(dataSource)
      setTree(schema.sections)
    }
  }, [id])

  useEffect(() => {
    load()
  }, [load])

  // poll a running job
  useEffect(() => {
    if (!job || ['COMPLETED', 'FAILED'].includes(job.status)) return
    const timer = setInterval(async () => {
      const updated = await api.job(job.id)
      setJob(updated)
      if (['COMPLETED', 'FAILED'].includes(updated.status)) load()
    }, 1500)
    return () => clearInterval(timer)
  }, [job, load])

  if (!config) return <Spinner />

  const allTables = Object.fromEntries(
    (source?.tables || []).map((table) => [table.name, table.columns.map((c) => c.name)]),
  )

  async function save() {
    setBusy(true)
    try {
      await api.updateConfiguration(id, { data_args: dataArgs, training_params: training })
      setMessage({ tone: 'success', text: 'Configuration saved.' })
    } catch (error) {
      setMessage({ tone: 'danger', text: error.message })
    } finally {
      setBusy(false)
    }
  }

  async function autoconfigure({ requireCatalog = true } = {}) {
    setBusy(true)
    setCatalogGap(false)
    try {
      const result = await api.autoconfigure(id, { requireCatalog })
      // Say which source this actually came from. "Derived from Atlas" when the
      // catalog knew nothing about these tables is the misreading this screen
      // has to prevent.
      const engine = `Engine: ${result.engine}.`
      const order = `Order: ${result.order.join(' → ')}`
      if (result.derived_from_catalog) {
        setMessage({
          tone: 'success',
          text: `Derived from your catalog (${result.atlas_source}). ${engine} ${order}`,
        })
      } else {
        const coverage = result.catalog_coverage || {}
        const missing = [
          ...(coverage.tables_missing || []),
          ...(coverage.tables_ungoverned || []),
        ]
        // Two different situations, and conflating them would misdescribe the
        // governance position: either nothing knows about these tables, or the
        // only tags available are the bundled illustrative ones.
        const why = missing.length
          ? `These tables are not in the catalog (${missing.join(', ')}), so keys come ` +
            `from profiling and protection from name and value inference.`
          : `The only classifications available are the bundled demo fixture, which is ` +
            `ILLUSTRATIVE sample data rather than your catalog.`
        setMessage({
          tone: 'warn',
          text: `Not derived from your catalog. ${why} Review the Governance screen before releasing anything. ${engine} ${order}`,
        })
      }
      await load()
    } catch (error) {
      // 409 means the catalog does not cover these tables. That is recoverable,
      // so offer the fallback rather than leaving a dead end.
      if (error.status === 409) setCatalogGap(true)
      setMessage({ tone: 'danger', text: error.message })
    } finally {
      setBusy(false)
    }
  }

  async function startTraining() {
    setBusy(true)
    try {
      await api.updateConfiguration(id, { data_args: dataArgs, training_params: training })
      const started = await api.train(id, { model_name: `${config.name}-model` })
      setJob({ id: started.job_id, status: 'QUEUED', progress: 0 })
    } catch (error) {
      setMessage({ tone: 'danger', text: error.message })
    } finally {
      setBusy(false)
    }
  }

  // Each engine reads its own section, so switching engines does not discard
  // the other's settings -- and, more importantly, the panel never offers a
  // parameter the selected engine ignores.
  const isARF = (config.model_type || 'SPN').toUpperCase().startsWith('ARF')
  const configKey = isARF ? 'arf_config' : 'spn_config'
  const spn = training[configKey] || {}
  function setSpn(key, value) {
    setTraining({ ...training, [configKey]: { ...spn, [key]: value } })
  }

  const rowCount = source?.tables?.reduce((total, table) => Math.max(total, table.rows), 0) || 0
  const betaWarning = !isARF && spn.beta >= rowCount && rowCount > 0

  return (
    <div>
      <div className="mb-6 flex items-start justify-between">
        <div>
          <h1 className="font-display text-2xl font-semibold text-fg">{config.name}</h1>
          <div className="mt-1 flex items-center gap-2 text-sm text-fg-subtle">
            <Status value={config.status} />
            <span>{config.model_type}</span>
            <span>·</span>
            <span>{config.selected_tables?.join(', ')}</span>
          </div>
        </div>
        <div className="flex gap-2">
          <button
            className="btn-ghost"
            onClick={() => autoconfigure()}
            disabled={busy}
            type="button"
          >
            <ShieldIcon size={16} />
            Auto-configure from Atlas
          </button>
          <button className="btn-ghost" onClick={save} disabled={busy} type="button">
            Save
          </button>
          <button className="btn-primary" onClick={startTraining} disabled={busy} type="button">
            <PlayIcon size={16} />
            Start Training
          </button>
        </div>
      </div>

      {message && <Banner tone={message.tone}>{message.text}</Banner>}

      {catalogGap && (
        <Banner tone="warn" title="No catalog entry for these tables">
          <div className="mb-3">
            Import your classifications to govern this dataset properly, or derive the
            schema from the data alone. Derived that way, keys come from profiling and
            protection from name and value inference, and every report will say so.
          </div>
          <button
            className="btn-ghost"
            onClick={() => autoconfigure({ requireCatalog: false })}
            disabled={busy}
            type="button"
          >
            Configure from the data instead
          </button>
        </Banner>
      )}

      {job && (
        <Banner tone={job.status === 'FAILED' ? 'danger' : 'info'} title={`Job ${job.status}`}>
          <div className="mb-2">{job.message}</div>
          <div className="h-1.5 w-full overflow-hidden rounded bg-surface/60">
            <div
              className="h-full bg-accent transition-all"
              style={{ width: `${(job.progress || 0) * 100}%` }}
            />
          </div>
          {job.status === 'COMPLETED' && job.result?.model_id && (
            <button
              type="button"
              className="mt-2 text-sm font-medium underline"
              onClick={() => navigate(`/app/models/${job.result.model_id}`)}
            >
              View fidelity &amp; privacy report →
            </button>
          )}
          {job.status === 'FAILED' && (
            <pre className="mt-2 max-h-40 overflow-auto text-xs">{job.result?.error}</pre>
          )}
        </Banner>
      )}

      <Section title="Basic Details">
        <div className="grid gap-4 md:grid-cols-4">
          <div>
            <div className="label">Model type</div>
            <div className="mt-1 text-sm text-fg-muted">{config.model_type}</div>
          </div>
          <div>
            <div className="label">Dataset</div>
            <div className="mt-1 text-sm text-fg-muted">{source?.name || '—'}</div>
          </div>
          <div>
            <div className="label">Device</div>
            <div className="mt-1 text-sm text-fg-muted">{config.device_type}</div>
          </div>
          <div>
            <div className="label">Compute</div>
            <div className="mt-1 text-sm text-fg-muted">
              {config.vcpu} vCPU · {config.ram_gb} GB · {config.gpu} GPU
            </div>
          </div>
        </div>
      </Section>

      <Section
        title="Data Arguments"
        description="Rendered live from schema.yaml — the same parameter tree the vendor UI exposes."
        right={
          <div className="flex items-center gap-3">
            <div className="flex rounded-md border border-line p-0.5">
              {['high', 'all'].map((mode) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => setFilter(mode)}
                  className={`rounded px-3 py-1 text-xs font-medium ${
                    filter === mode ? 'bg-accent text-accent-fg shadow-sm' : 'text-fg-muted'
                  }`}
                >
                  {mode === 'high' ? 'High Impact' : 'All'}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-2 text-xs text-fg-subtle">
              UI
              <Toggle checked={jsonView} onChange={setJsonView} />
              JSON
            </div>
          </div>
        }
      >
        {!tree ? (
          <Spinner label="Loading parameter schema…" />
        ) : jsonView ? (
          <textarea
            className="input h-[420px] font-mono text-xs"
            value={JSON.stringify(dataArgs, null, 2)}
            onChange={(event) => {
              try {
                setDataArgs(JSON.parse(event.target.value))
              } catch {
                /* keep typing */
              }
            }}
          />
        ) : (
          <FieldTree
            sections={tree}
            values={dataArgs}
            onChange={setDataArgs}
            allTables={allTables}
            filter={filter}
          />
        )}
      </Section>

      <Section
        title="Training Parameters"
        description={isARF ? 'ARF_Config' : 'SPN_Config'}
      >
        {isARF ? (
          <>
            <Banner tone="warn" title="No differential privacy on this engine">
              ARF learns split thresholds rather than counts, so the SPN&apos;s epsilon
              accounting does not transfer. If Apache Atlas marks a selected column
              SENSITIVE, this run will still produce a model that is <strong>not</strong>{' '}
              private, and the report will say so. Switch the configuration to SPN for
              sensitive data.
            </Banner>
            <div className="grid gap-4 md:grid-cols-4">
              <div>
                <div className="label">n_trees</div>
                <input
                  type="number"
                  className="input mt-1"
                  value={spn.n_trees ?? 60}
                  onChange={(event) => setSpn('n_trees', Number(event.target.value))}
                />
                <p className="mt-1 text-xs text-fg-subtle">
                  Forest size. More trees give a smoother density and cost linear time.
                </p>
              </div>
              <div>
                <div className="label">max_rounds</div>
                <input
                  type="number"
                  className="input mt-1"
                  value={spn.max_rounds ?? 5}
                  onChange={(event) => setSpn('max_rounds', Number(event.target.value))}
                />
                <p className="mt-1 text-xs text-fg-subtle">
                  Adversarial rounds. Training stops early once the discriminator is at
                  chance, so this is a ceiling rather than a target.
                </p>
              </div>
              <div>
                <div className="label">min_node_size</div>
                <input
                  type="number"
                  className="input mt-1"
                  value={spn.min_node_size ?? 20}
                  onChange={(event) => setSpn('min_node_size', Number(event.target.value))}
                />
                <p className="mt-1 text-xs text-fg-subtle">
                  Rows per leaf. Drives both resolution and cost: raising it from 20 to 60
                  cut generation time roughly threefold here with no measurable quality
                  loss.
                </p>
              </div>
              <div>
                <div className="label">delta</div>
                <input
                  type="number"
                  step="0.01"
                  className="input mt-1"
                  value={spn.delta ?? 0.05}
                  onChange={(event) => setSpn('delta', Number(event.target.value))}
                />
                <p className="mt-1 text-xs text-fg-subtle">
                  Convergence tolerance. Training stops when the discriminator&apos;s
                  accuracy is within delta of 0.5.
                </p>
              </div>
            </div>
          </>
        ) : (
          <>
            {betaWarning && (
              <Banner tone="warn" title="beta is larger than the table">
                With beta ({Number(spn.beta).toLocaleString()}) at or above the row count (
                {rowCount.toLocaleString()}), no SUM node can form and every column is
                modelled independently — correlations will be lost. Lower beta well below
                the row count.
              </Banner>
            )}
            <div className="grid gap-4 md:grid-cols-3">
              <div>
                <div className="label">beta</div>
                <input
                  type="number"
                  className="input mt-1"
                  value={spn.beta ?? 100000}
                  onChange={(event) => setSpn('beta', Number(event.target.value))}
                />
                <p className="mt-1 text-xs text-fg-subtle">
                  Minimum leaf size to build a SUM node. Nodes with less data become
                  PRODUCT nodes. Keep it well below the row count.
                </p>
              </div>
              <div>
                <div className="label">private</div>
                <div className="mt-2">
                  <Toggle
                    checked={Boolean(spn.private)}
                    onChange={(value) => setSpn('private', value)}
                    label="Differential privacy"
                  />
                </div>
                <p className="mt-1 text-xs text-fg-subtle">
                  Differential privacy. Forced on automatically when Atlas marks a selected
                  column SENSITIVE.
                </p>
              </div>
              {spn.private && (
                <div>
                  <div className="label">epsilon</div>
                  <input
                    type="number"
                    step="0.1"
                    className="input mt-1"
                    value={spn.epsilon ?? 2.0}
                    onChange={(event) => setSpn('epsilon', Number(event.target.value))}
                  />
                  <p className="mt-1 text-xs text-fg-subtle">
                    Lower is more private. Unconditional only when the column domains are
                    declared.
                  </p>
                </div>
              )}
            </div>
          </>
        )}
      </Section>
    </div>
  )
}
