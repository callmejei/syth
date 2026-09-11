import { useEffect, useRef, useState } from 'react'
import { DownloadIcon } from './icons'

/**
 * Format picker for a download.
 *
 * CSV and parquet are both offered because they serve different readers: CSV
 * opens in Excel, which is what a reviewer will do, while parquet preserves
 * dtypes for anything downstream.
 */
export default function DownloadMenu({ onDownload, label = 'Download', title, disabled }) {
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const ref = useRef(null)

  useEffect(() => {
    if (!open) return undefined
    function onDocument(event) {
      if (!ref.current?.contains(event.target)) setOpen(false)
    }
    function onKey(event) {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDocument)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDocument)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  async function pick(fmt) {
    setBusy(true)
    setError(null)
    try {
      await onDownload(fmt)
      setOpen(false)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="relative inline-block" ref={ref}>
      <button
        className="btn-link px-1.5"
        type="button"
        disabled={disabled || busy}
        title={title}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <DownloadIcon size={14} />
        {busy ? 'Preparing…' : label}
      </button>

      {open && (
        <div
          role="menu"
          className="absolute right-0 z-30 mt-1 w-44 overflow-hidden rounded-lg border border-line bg-surface-raised shadow-lg"
        >
          <button
            role="menuitem"
            type="button"
            className="block w-full px-3 py-2 text-left text-sm text-fg transition hover:bg-sunken"
            onClick={() => pick('csv')}
          >
            CSV
            <span className="block text-xs text-fg-subtle">Opens in Excel</span>
          </button>
          <button
            role="menuitem"
            type="button"
            className="block w-full border-t border-line px-3 py-2 text-left text-sm text-fg transition hover:bg-sunken"
            onClick={() => pick('parquet')}
          >
            Parquet
            <span className="block text-xs text-fg-subtle">Keeps column types</span>
          </button>
        </div>
      )}

      {error && (
        <p role="alert" className="absolute right-0 top-full mt-1 w-64 text-xs text-danger-text">
          {error}
        </p>
      )}
    </div>
  )
}
