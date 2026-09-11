import { useEffect, useState } from 'react'
import api from '../api'
import { Banner, Modal, Spinner } from './ui'

/**
 * Delete confirmation for a dataset.
 *
 * The dialog loads what depends on the dataset before offering the button, so
 * the consequence is stated up front rather than surfacing as a 409 after the
 * user has already committed. Cascade is a separate, explicit opt-in.
 */
export default function ConfirmDelete({ source, onClose, onDeleted }) {
  const [usage, setUsage] = useState(null)
  const [cascade, setCascade] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    api
      .dataSourceUsage(source.id)
      .then(setUsage)
      .catch((err) => setError(err.message))
  }, [source.id])

  const blocked = usage && (usage.configurations.length > 0 || usage.models.length > 0)

  async function remove() {
    setBusy(true)
    setError(null)
    try {
      const result = await api.deleteDataSource(source.id, cascade)
      onDeleted(result)
    } catch (err) {
      setError(err.message)
      setBusy(false)
    }
  }

  return (
    <Modal
      title={`Delete "${source.name}"?`}
      description="This removes the database record and the files on disk. It cannot be undone."
      onClose={onClose}
      footer={
        <>
          <button className="btn-ghost" type="button" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button
            className="btn-danger"
            type="button"
            onClick={remove}
            disabled={busy || !usage || (blocked && !cascade)}
          >
            {busy ? 'Deleting…' : blocked && cascade ? 'Delete everything' : 'Delete dataset'}
          </button>
        </>
      }
    >
      {error && <Banner tone="danger">{error}</Banner>}
      {!usage && !error && <Spinner rows={2} label="Checking what uses this dataset" />}

      {usage && !blocked && (
        <p className="text-sm text-fg-muted">
          Nothing depends on this dataset, so it is safe to remove.
        </p>
      )}

      {usage && blocked && (
        <div className="space-y-3">
          <Banner tone="warn" title="Other records depend on this dataset">
            Deleting it will leave them without their input data.
          </Banner>

          {usage.configurations.length > 0 && (
            <div>
              <div className="label">Configurations ({usage.configurations.length})</div>
              <ul className="mt-1 space-y-0.5 text-sm text-fg-muted">
                {usage.configurations.map((item) => (
                  <li key={item.id}>{item.name}</li>
                ))}
              </ul>
            </div>
          )}

          {usage.models.length > 0 && (
            <div>
              <div className="label">Trained models ({usage.models.length})</div>
              <ul className="mt-1 space-y-0.5 text-sm text-fg-muted">
                {usage.models.map((item) => (
                  <li key={item.id}>{item.name}</li>
                ))}
              </ul>
            </div>
          )}

          <label className="flex items-start gap-2 rounded-md border border-danger/30 bg-danger-subtle/40 p-3 text-sm">
            <input
              type="checkbox"
              className="mt-0.5"
              checked={cascade}
              onChange={(event) => setCascade(event.target.checked)}
            />
            <span className="text-fg">
              Also delete the configurations, models and jobs listed above, along with
              any synthetic data they generated.
            </span>
          </label>
        </div>
      )}
    </Modal>
  )
}
