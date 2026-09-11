import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import api from '../api'
import { Banner, Metric, PageHeader, Section, Spinner, Table } from '../components/ui'
import DownloadMenu from '../components/DownloadMenu'

function GeneratePanel({ modelId, tables }) {
  const [rows, setRows] = useState(2000)
  const [job, setJob] = useState(null)
  const [preview, setPreview] = useState(null)
  const rootTable = tables[0]

  useEffect(() => {
    if (!job || ['COMPLETED', 'FAILED'].includes(job.status)) return
    const timer = setInterval(async () => {
      const updated = await api.job(job.id)
      setJob(updated)
      if (updated.status === 'COMPLETED') {
        api.preview(updated.id, rootTable, 8).then(setPreview).catch(() => {})
      }
    }, 1500)
    return () => clearInterval(timer)
  }, [job, rootTable])

  async function start() {
    const started = await api.generate({
      model_id: modelId,
      n_rows: { [rootTable]: Number(rows) },
      seed: 42,
    })
    setJob({ id: started.job_id, status: 'QUEUED', progress: 0 })
    setPreview(null)
  }

  return (
    <Section title="Generate synthetic data">
      <div className="mb-4 flex items-end gap-3">
        <div>
          <div className="label">Rows for {rootTable}</div>
          <input
            type="number"
            className="input mt-1 w-40"
            value={rows}
            onChange={(event) => setRows(event.target.value)}
          />
          <p className="mt-1 text-xs text-fg-subtle">
            Child tables scale automatically from the learned cardinality.
          </p>
        </div>
        <button className="btn-primary" onClick={start} type="button">
          Generate
        </button>
      </div>

      {job && (
        <Banner tone={job.status === 'FAILED' ? 'danger' : 'info'} title={`Job ${job.status}`}>
          {job.message}
          {job.status === 'COMPLETED' && job.result?.files && (
            <>
              <ul className="mt-2 space-y-1 text-xs">
                {job.result.files.map((file) => (
                  <li key={file.table} className="flex items-center justify-between gap-3">
                    <span>
                      {file.table}: {file.rows.toLocaleString()} rows ·{' '}
                      {(file.bytes / 1024).toFixed(0)} KB
                    </span>
                    <DownloadMenu
                      label="Download"
                      onDownload={(fmt) =>
                        api.downloadOutput(job.id, { table: file.table, fmt })
                      }
                    />
                  </li>
                ))}
              </ul>
              {job.result.files.length > 1 && (
                <div className="mt-2 border-t border-line pt-2">
                  <DownloadMenu
                    label="Download all tables (zip)"
                    onDownload={(fmt) => api.downloadOutput(job.id, { fmt })}
                  />
                </div>
              )}
            </>
          )}
        </Banner>
      )}

      {preview && (
        <div className="overflow-x-auto rounded-md border border-line">
          <table className="w-full text-xs">
            <thead className="bg-sunken">
              <tr>
                {preview.columns.map((column) => (
                  <th key={column} className="px-2 py-1.5 text-left font-medium text-fg-muted">
                    {column}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.rows.map((row, index) => (
                <tr key={index} className="border-t border-line">
                  {preview.columns.map((column) => (
                    <td key={column} className="whitespace-nowrap px-2 py-1.5 text-fg-muted">
                      {String(row[column]).slice(0, 28)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Section>
  )
}

function UtilityPanel({ utility }) {
  const ratio = utility.mean_utility_ratio
  const auc = utility.mean_detection_auc

  return (
    <Section
      title="Utility evaluation"
      description="Train on synthetic, test on real — does this data actually work as a substitute?"
    >
      <div className="mb-4 grid grid-cols-2 gap-4 lg:grid-cols-3">
        <Metric
          label="Utility ratio"
          value={ratio != null ? `${(ratio * 100).toFixed(0)}%` : '—'}
          tone={ratio > 0.9 ? 'text-success-text' : ratio > 0.75 ? 'text-warning-text' : 'text-danger-text'}
          hint="model trained on synthetic vs trained on real, both scored on real"
        />
        <Metric
          label="Detection AUC"
          value={auc != null ? auc.toFixed(3) : '—'}
          tone={auc != null && auc < 0.6 ? 'text-success-text' : 'text-warning-text'}
          hint="0.5 = indistinguishable from real"
        />
        <Metric label="Targets scored" value={utility.targets_evaluated ?? 0} />
      </div>

      {Object.entries(utility.tables || {}).map(([name, entry]) => {
        if (!entry.tstr) return null
        const detection = entry.detection || {}
        return (
          <div key={name} className="mb-4 rounded-md border border-line p-4">
            <div className="mb-2 flex items-center justify-between">
              <span className="font-medium text-fg">{name}</span>
              {detection.available && (
                <span
                  className={`pill ${
                    detection.detection_auc < 0.6
                      ? 'bg-success-subtle text-success-text'
                      : 'bg-warning-subtle text-warning-text'
                  }`}
                >
                  detection AUC {detection.detection_auc}
                </span>
              )}
            </div>

            {entry.tstr.length > 0 ? (
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-line text-left text-[11px] uppercase text-fg-subtle">
                    <th className="px-2 py-1 font-medium">Target</th>
                    <th className="px-2 py-1 font-medium">Metric</th>
                    <th className="px-2 py-1 font-medium">Trained on real</th>
                    <th className="px-2 py-1 font-medium">Trained on synthetic</th>
                    <th className="px-2 py-1 font-medium">Ratio</th>
                  </tr>
                </thead>
                <tbody>
                  {entry.tstr.map((row) => (
                    <tr key={row.target} className="border-b border-line">
                      <td className="px-2 py-1.5 font-medium">{row.target}</td>
                      <td className="px-2 py-1.5 text-fg-subtle">{row.metric}</td>
                      <td className="px-2 py-1.5">{row.train_on_real}</td>
                      <td className="px-2 py-1.5">{row.train_on_synthetic}</td>
                      <td
                        className={`px-2 py-1.5 font-medium ${
                          row.utility_ratio > 0.9
                            ? 'text-success-text'
                            : row.utility_ratio > 0.75
                              ? 'text-warning-text'
                              : 'text-danger-text'
                        }`}
                      >
                        {(row.utility_ratio * 100).toFixed(0)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="text-xs text-fg-subtle">
                No target in this table is predictable from the real data, so there is
                nothing here that can measure utility.
              </p>
            )}

            {entry.not_measurable?.length > 0 && (
              <p className="mt-2 text-xs text-fg-subtle">
                Not measurable (no signal in the real data):{' '}
                {entry.not_measurable.map((row) => row.target).join(', ')}
              </p>
            )}

            {detection.available && detection.detection_auc > 0.65 && (
              <p className="mt-2 text-xs text-warning-text">
                Distinguishable from real data. Strongest tells:{' '}
                {detection.most_revealing_columns.map((c) => c.column).join(', ')}
              </p>
            )}
          </div>
        )
      })}

      <p className="text-xs text-fg-subtle">{utility.method}</p>
    </Section>
  )
}

export default function ModelReport() {
  const { id } = useParams()
  const [report, setReport] = useState(null)
  const [model, setModel] = useState(null)

  useEffect(() => {
    api.report(id).then(setReport)
    api.model(id).then(setModel)
  }, [id])

  if (!report || !model) return <Spinner />

  const tables = Object.keys(report.tables || {})
  const dp = report.differential_privacy || {}

  return (
    <div>
      <PageHeader title={model.name} subtitle="Fidelity and privacy report" />

      <div className="mb-5 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Metric
          label="Overall fidelity"
          value={`${(report.overall_fidelity * 100).toFixed(1)}%`}
          tone={report.overall_fidelity > 0.85 ? 'text-success-text' : 'text-warning-text'}
        />
        <Metric
          label="Differential privacy"
          value={dp.enabled ? 'On' : 'Off'}
          tone={dp.enabled ? 'text-violet-text' : 'text-fg'}
        />
        <Metric label="Tables" value={tables.length} />
        <Metric
          label="Warnings"
          value={report.warnings?.length || 0}
          tone={report.warnings?.length ? 'text-warning-text' : 'text-fg'}
        />
      </div>

      {report.warnings?.map((warning, index) => (
        <Banner key={index} tone="warn">
          {warning}
        </Banner>
      ))}

      {dp.enabled && (
        <Banner tone="info" title="Scope of the differential privacy guarantee">
          {dp.caveat}
        </Banner>
      )}

      {report.utility?.available && <UtilityPanel utility={report.utility} />}

      {tables.length > 0 && <GeneratePanel modelId={id} tables={tables} />}

      {Object.entries(report.tables || {}).map(([name, table]) => (
        <Section
          key={name}
          title={name}
          description={`${table.real_rows.toLocaleString()} real → ${table.synth_rows.toLocaleString()} synthetic · fidelity ${(
            table.fidelity_score * 100
          ).toFixed(1)}% over ${table.scored_columns} scored columns`}
        >
          {table.protected_columns?.length > 0 && (
            <p className="mb-3 text-xs text-fg-subtle">
              Protected (not scored, by design):{' '}
              {table.protected_columns.map((column) => (
                <span key={column} className="pill mr-1 bg-warning-subtle text-warning-text">
                  {column}
                </span>
              ))}
            </p>
          )}
          <Table
            columns={[
              { key: 'column', label: 'Column' },
              { key: 'kind', label: 'Kind' },
              {
                key: 'score',
                label: 'Score',
                render: (row) =>
                  row.scored && row.score != null ? (
                    <span
                      className={
                        row.score > 0.85
                          ? 'text-success-text'
                          : row.score > 0.7
                            ? 'text-warning-text'
                            : 'text-danger-text'
                      }
                    >
                      {(row.score * 100).toFixed(0)}%
                    </span>
                  ) : (
                    <span className="text-xs text-fg-subtle">not scored</span>
                  ),
              },
              {
                key: 'detail',
                label: 'Detail',
                render: (row) =>
                  row.kind === 'categorical' ? (
                    `TVD ${row.tvd?.toFixed(3)} · ${row.real_cardinality}→${row.synth_cardinality} categories`
                  ) : row.kind === 'protected' ? (
                    <span className={row.value_overlap ? 'text-danger-text' : 'text-success-text'}>
                      {row.value_overlap} real values reproduced
                    </span>
                  ) : row.real_median != null ? (
                    `median ${row.real_median} → ${row.synth_median}`
                  ) : (
                    <span className="text-xs text-fg-subtle">{row.note || ''}</span>
                  ),
              },
            ]}
            rows={table.columns.map((column, index) => ({ ...column, id: index }))}
          />
        </Section>
      ))}
    </div>
  )
}
