import { useEffect, useState } from 'react'
import api from '../api'
import { Banner, PageHeader, Section, Spinner, Toggle } from '../components/ui'

const TOGGLES = [
  ['enabled', 'Enabled'],
  ['show_progress', 'Show progress'],
  ['quick_train', 'Quick train'],
  ['default_gpu', 'Default GPU'],
  ['parquet_dataset', 'Parquet dataset'],
  ['is_beta', 'Is beta'],
  ['multi_table', 'Multi table'],
]

export default function ModelTypes() {
  const [rows, setRows] = useState(null)
  const [error, setError] = useState(null)

  function load() {
    api.modelTypes().then(setRows)
  }
  useEffect(load, [])

  async function toggle(row, key, value) {
    setError(null)
    try {
      await api.updateModelType(row.id, { [key]: value })
      load()
    } catch (err) {
      setError(err.message)
    }
  }

  if (!rows) return <Spinner />

  return (
    <div>
      <PageHeader title="Model types" subtitle="Modify and rename deep generative models" />
      {error && <Banner tone="danger">{error}</Banner>}

      <Banner tone="info" title="Why only one engine is enabled">
        SPN is implemented in-house on numpy/scipy/scikit-learn, so nothing here restricts
        production use. The GAN and VAE engines are left disabled because their reference
        implementations ship under the Business Source Licence, which requires a paid commercial
        licence in production — that would reintroduce exactly the vendor cost this build exists to
        avoid.
      </Banner>

      <Section title={`${rows.length} model type(s)`}>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-line text-left text-[11px] uppercase tracking-wide text-fg-subtle">
                <th className="px-3 py-2 font-medium">Model name</th>
                <th className="px-3 py-2 font-medium">Description</th>
                {TOGGLES.map(([key, label]) => (
                  <th key={key} className="px-3 py-2 text-center font-medium">
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.id} className="border-b border-line align-top">
                  <td className="px-3 py-3">
                    <div className="font-medium text-fg">{row.name}</div>
                    <span className="pill mt-1 bg-sunken text-fg-muted">{row.code}</span>
                  </td>
                  <td className="max-w-sm px-3 py-3">
                    <div className="text-fg-muted">{row.description}</div>
                    {row.licence_note && (
                      <div
                        className={`mt-1 text-xs ${
                          row.licence_note.startsWith('NOT ENABLED')
                            ? 'text-danger-text'
                            : 'text-success-text'
                        }`}
                      >
                        {row.licence_note}
                      </div>
                    )}
                  </td>
                  {TOGGLES.map(([key]) => (
                    <td key={key} className="px-3 py-3 text-center">
                      <div className="flex justify-center">
                        <Toggle
                          checked={row[key]}
                          disabled={key === 'enabled' && row.licence_note.startsWith('NOT ENABLED')}
                          title={
                            key === 'enabled' && row.licence_note.startsWith('NOT ENABLED')
                              ? row.licence_note
                              : undefined
                          }
                          onChange={(value) => toggle(row, key, value)}
                        />
                      </div>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  )
}
