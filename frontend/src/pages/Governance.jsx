import { useEffect, useState } from 'react'
import api from '../api'
import { Banner, Metric, PageHeader, Section, Spinner } from '../components/ui'

const STRATEGY_STYLES = {
  SUPPRESS: 'bg-danger-subtle text-danger-text',
  PSEUDONYM: 'bg-warning-subtle text-warning-text',
  GENERALISE: 'bg-info-subtle text-info-text',
  MODEL_DP: 'bg-violet-subtle text-violet-text',
  MODEL: 'bg-success-subtle text-success-text',
}

/** Where the classifications came from. The distinction matters: a policy built
 *  from demo tags must never be mistaken for one built from your catalog. */
function SourceBanner({ policy, atlas }) {
  const source = policy?.source

  if (source === 'atlas') {
    return (
      <Banner tone="success" title="Live Apache Atlas">
        Classifications read from Atlas at {atlas?.base_url}. Decisions below reflect your
        catalog as it stands right now.
      </Banner>
    )
  }

  if (source === 'local-catalog') {
    return (
      <Banner tone="success" title="Imported catalog">
        Using classifications imported from your own catalog export — these are your tags,
        not sample data. {policy.detail}
      </Banner>
    )
  }

  return (
    <Banner tone="warn" title="Demo classifications — not your catalog">
      No live Atlas and no imported catalog, so the bundled sample tags are being used.
      Every decision below is illustrative only.
      {atlas && !atlas.reachable && <> Atlas is unreachable ({atlas.reason}).</>}
      <div className="mt-2">
        To govern this with your real metadata without any network access, export the
        classifications from your Atlas (or get a spreadsheet from the governance team) and
        import them:
        <code className="mt-1 block rounded bg-surface/60 px-2 py-1 font-mono text-xs">
          docker compose exec backend python -m scripts.import_classifications --input
          /data/catalog/your-export.json
        </code>
      </div>
    </Banner>
  )
}

export default function Governance() {
  const [sources, setSources] = useState([])
  const [sourceId, setSourceId] = useState('')
  const [mode, setMode] = useState('balanced')
  const [policy, setPolicy] = useState(null)
  const [atlas, setAtlas] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    api.dataSources().then((rows) => {
      setSources(rows)
      if (rows.length) setSourceId(rows[0].id)
    })
    api.atlasStatus().then(setAtlas)
  }, [])

  useEffect(() => {
    if (!sourceId) return
    setLoading(true)
    api
      .policy(sourceId, mode)
      .then(setPolicy)
      .finally(() => setLoading(false))
  }, [sourceId, mode])

  return (
    <div>
      <PageHeader
        title="Data Protection Policy"
        subtitle="Derived from Apache Atlas classifications — not guessed from the data."
      />

      <SourceBanner policy={policy} atlas={atlas} />

      <div className="mb-5 flex flex-wrap items-end gap-4">
        <div>
          <div className="label">Data source</div>
          <select
            className="input mt-1 min-w-[220px]"
            value={sourceId}
            onChange={(event) => setSourceId(event.target.value)}
          >
            {sources.map((source) => (
              <option key={source.id} value={source.id}>
                {source.name}
              </option>
            ))}
          </select>
        </div>
        <div>
          <div className="label">Policy mode</div>
          <select
            className="input mt-1"
            value={mode}
            onChange={(event) => setMode(event.target.value)}
          >
            <option value="permissive">Permissive</option>
            <option value="balanced">Balanced</option>
            <option value="strict">Strict — fail closed on untagged columns</option>
          </select>
        </div>
      </div>

      {loading && <Spinner />}

      {policy && !loading && (
        <>
          <div className="mb-5 grid grid-cols-2 gap-4 lg:grid-cols-4">
            <Metric label="Source" value={policy.source} hint={policy.detail} />
            <Metric
              label="Suppressed"
              value={policy.suppressed_total}
              tone="text-danger-text"
              hint="dropped entirely"
            />
            <Metric
              label="Pseudonymised"
              value={policy.pseudonymised_total}
              tone="text-warning-text"
              hint="replaced, never modelled"
            />
            <Metric
              label="DP required"
              value={policy.requires_dp ? 'Yes' : 'No'}
              tone={policy.requires_dp ? 'text-violet-text' : 'text-fg'}
              hint={policy.requires_dp ? 'sensitive columns present' : ''}
            />
          </div>

          {policy.requires_dp && (
            <Banner tone="info" title="Differential privacy will be enforced">
              At least one selected column is classified SENSITIVE in Atlas, so training runs with
              differential privacy whether or not the user ticks the box. This is the control that
              a reviewer can trace back to a catalog entry.
            </Banner>
          )}

          {Object.entries(policy.tables).map(([table, entry]) => (
            <Section key={table} title={table}>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-line text-left text-[11px] uppercase tracking-wide text-fg-subtle">
                    <th className="px-3 py-2 font-medium">Column</th>
                    <th className="px-3 py-2 font-medium">Atlas classifications</th>
                    <th className="px-3 py-2 font-medium">Strategy</th>
                    <th className="px-3 py-2 font-medium">Why</th>
                  </tr>
                </thead>
                <tbody>
                  {entry.decisions.map((decision) => (
                    <tr key={decision.column} className="border-b border-line">
                      <td className="px-3 py-2 font-medium text-fg">{decision.column}</td>
                      <td className="px-3 py-2">
                        <div className="flex flex-wrap gap-1">
                          {decision.classifications.length ? (
                            decision.classifications.map((tag) => (
                              <span key={tag} className="pill bg-sunken text-fg-muted">
                                {tag}
                              </span>
                            ))
                          ) : (
                            <span className="text-xs text-fg-subtle">untagged</span>
                          )}
                        </div>
                      </td>
                      <td className="px-3 py-2">
                        <span
                          className={`pill ${STRATEGY_STYLES[decision.strategy] || 'bg-sunken'}`}
                        >
                          {decision.strategy}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-xs text-fg-subtle">{decision.reason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Section>
          ))}
        </>
      )}
    </div>
  )
}
