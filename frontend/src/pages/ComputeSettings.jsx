import { useEffect, useState } from 'react'
import api from '../api'
import { Banner, PageHeader, Section, Spinner } from '../components/ui'

export default function ComputeSettings() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [draft, setDraft] = useState({})

  function load() {
    api.compute().then((result) => {
      setData(result)
      setDraft(
        Object.fromEntries(
          result.profiles.map((profile) => [
            profile.id,
            { vcpu: profile.vcpu, ram_gb: profile.ram_gb, gpu: profile.gpu },
          ]),
        ),
      )
    })
  }
  useEffect(load, [])

  async function save(profileId) {
    setError(null)
    try {
      await api.updateProfile(profileId, draft[profileId])
      load()
    } catch (err) {
      setError(err.message)
    }
  }

  if (!data) return <Spinner />
  const node = data.nodes[0]

  return (
    <div>
      <PageHeader title="Compute Settings" subtitle="Compute profiles and node limits" />
      {error && <Banner tone="danger">{error}</Banner>}

      {node && (
        <Banner tone="info">
          Profiles are capped by <strong>{node.name}</strong>: {node.max_vcpu} vCPU ·{' '}
          {node.max_ram_gb} GB RAM · {node.max_gpu} GPU. Requests above these limits are rejected.
        </Banner>
      )}

      <div className="grid gap-4 md:grid-cols-3">
        {data.profiles.map((profile) => (
          <Section key={profile.id} title={profile.name} description="Compute profile configuration">
            <div className="space-y-3">
              {[
                ['vcpu', 'vCPU', node?.max_vcpu],
                ['ram_gb', 'RAM (GB)', node?.max_ram_gb],
                ['gpu', 'GPU', node?.max_gpu],
              ].map(([key, label, max]) => (
                <div key={key}>
                  <div className="label">{label}</div>
                  <input
                    type="number"
                    className="input mt-1"
                    value={draft[profile.id]?.[key] ?? ''}
                    onChange={(event) =>
                      setDraft({
                        ...draft,
                        [profile.id]: {
                          ...draft[profile.id],
                          [key]: Number(event.target.value),
                        },
                      })
                    }
                  />
                  <div className="mt-0.5 text-xs text-fg-subtle">Max: {max}</div>
                </div>
              ))}
              <button className="btn-primary w-full" onClick={() => save(profile.id)} type="button">
                Save
              </button>
            </div>
          </Section>
        ))}
      </div>
    </div>
  )
}
