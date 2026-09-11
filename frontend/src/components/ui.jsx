export function PageHeader({ title, subtitle, actions }) {
  // One header shape for every surface: title and subtitle share a left edge,
  // actions align to the same baseline as the title, and the bottom margin is
  // the same everywhere so pages do not drift relative to each other.
  return (
    <div className="mb-7 flex flex-wrap items-start justify-between gap-4 border-b border-line pb-5">
      <div>
        <h1 className="font-display text-2xl font-semibold tracking-tight text-fg">{title}</h1>
        {subtitle && <p className="mt-1 max-w-2xl text-sm text-fg-muted">{subtitle}</p>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  )
}

export function Section({ title, description, children, right }) {
  return (
    <section className="card mb-5 animate-fade-in">
      <div className="flex items-center justify-between gap-4 border-b border-line px-5 py-3.5">
        <div>
          <h2 className="font-display text-sm font-semibold text-fg">{title}</h2>
          {description && <p className="mt-0.5 text-xs text-fg-muted">{description}</p>}
        </div>
        {right}
      </div>
      <div className="p-5">{children}</div>
    </section>
  )
}

export function Table({ columns, rows, empty = 'Nothing here yet.', caption }) {
  if (!rows.length) {
    return (
      <div className="rounded-lg border border-dashed border-line-strong py-10 text-center">
        <p className="text-sm text-fg-muted">{empty}</p>
      </div>
    )
  }
  return (
    <div className="-mx-2 overflow-x-auto">
      <table className="w-full text-sm">
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead>
          <tr className="border-b border-line text-left text-2xs uppercase tracking-wider text-fg-subtle">
            {columns.map((column) => (
              <th
                key={column.key}
                scope="col"
                className={`whitespace-nowrap px-3 py-2.5 font-semibold ${
                  column.numeric ? 'text-right' : ''
                }`}
              >
                {column.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr
              key={row.id || index}
              className="border-b border-line/60 transition-colors last:border-0 hover:bg-sunken"
            >
              {columns.map((column) => (
                <td
                  key={column.key}
                  className={`whitespace-nowrap px-3 py-3 text-fg-muted ${
                    column.numeric ? 'num text-right' : ''
                  }`}
                >
                  {column.render ? column.render(row) : row[column.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const STATUS_STYLES = {
  Ready: 'bg-success-subtle text-success-text',
  Completed: 'bg-success-subtle text-success-text',
  COMPLETED: 'bg-success-subtle text-success-text',
  Draft: 'bg-warning-subtle text-warning-text',
  QUEUED: 'bg-sunken text-fg-muted',
  RUNNING: 'bg-info-subtle text-info-text',
  FAILED: 'bg-danger-subtle text-danger-text',
}

export function Status({ value }) {
  const running = value === 'RUNNING'
  return (
    <span className={`pill ${STATUS_STYLES[value] || 'bg-sunken text-fg-muted'}`}>
      <span
        aria-hidden="true"
        className={`h-1.5 w-1.5 rounded-full bg-current ${running ? 'animate-pulse' : ''}`}
      />
      {value}
    </span>
  )
}

export function Impact({ tags = [] }) {
  const tag = tags.find((t) => t.includes('Impact'))
  if (!tag) return null
  const style = tag.startsWith('High')
    ? 'border-accent/40 text-accent-text bg-accent-subtle'
    : tag.startsWith('Medium')
      ? 'border-warning/40 text-warning-text bg-warning-subtle'
      : 'border-line-strong text-fg-subtle'
  return <span className={`pill border ${style}`}>{tag}</span>
}

export function Toggle({ checked, onChange, disabled, label, title }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={Boolean(checked)}
      aria-label={label}
      title={title || label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={`relative h-6 w-11 shrink-0 rounded-full transition-colors duration-200 ${
        checked ? 'bg-accent' : 'bg-line-strong'
      } ${disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}`}
    >
      <span
        className={`absolute top-1 h-4 w-4 rounded-full bg-surface shadow transition-all duration-200 ease-out ${
          checked ? 'left-6' : 'left-1'
        }`}
      />
    </button>
  )
}

const BANNER_TONES = {
  info: 'border-info/30 bg-info-subtle text-info-text',
  warn: 'border-warning/30 bg-warning-subtle text-warning-text',
  danger: 'border-danger/30 bg-danger-subtle text-danger-text',
  success: 'border-success/30 bg-success-subtle text-success-text',
}

const BANNER_ICONS = {
  info: <path d="M12 16v-5M12 8h.01M12 21a9 9 0 1 1 0-18 9 9 0 0 1 0 18z" />,
  warn: <path d="M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" />,
  danger: <path d="M12 8v5M12 17h.01M12 21a9 9 0 1 1 0-18 9 9 0 0 1 0 18z" />,
  success: <path d="m8 12 3 3 5-6M12 21a9 9 0 1 1 0-18 9 9 0 0 1 0 18z" />,
}

export function Banner({ tone = 'info', title, children }) {
  const assertive = tone === 'danger'
  return (
    <div
      role={assertive ? 'alert' : 'status'}
      aria-live={assertive ? 'assertive' : 'polite'}
      className={`mb-4 flex animate-fade-in gap-3 rounded-xl border px-4 py-3 text-sm ${BANNER_TONES[tone]}`}
    >
      <svg
        width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
        strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round"
        className="mt-0.5 shrink-0" aria-hidden="true" focusable="false"
      >
        {BANNER_ICONS[tone]}
      </svg>
      <div className="min-w-0">
        {title && <div className="font-semibold">{title}</div>}
        <div className={title ? 'mt-0.5' : ''}>{children}</div>
      </div>
    </div>
  )
}

/** Skeleton, not a spinner: it occupies the layout the content will take, so
 *  nothing jumps when data lands. */
export function Spinner({ label = 'Loading', rows = 3 }) {
  return (
    <div role="status" aria-live="polite" className="space-y-3 py-4">
      <span className="sr-only">{label}</span>
      {Array.from({ length: rows }).map((_, index) => (
        <div key={index} className="flex gap-3" aria-hidden="true">
          {['w-1/4', 'flex-1', 'w-16'].map((width) => (
            <div
              key={width}
              className={`relative h-4 overflow-hidden rounded bg-sunken ${width}`}
            >
              <div className="absolute inset-0 -translate-x-full animate-shimmer bg-gradient-to-r from-transparent via-fg/[0.06] to-transparent" />
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}

export function Metric({ label, value, hint, tone, icon }) {
  return (
    <div className="card p-4 transition-shadow hover:shadow-md">
      <div className="flex items-start justify-between gap-2">
        <div className="text-2xs font-medium uppercase tracking-wider text-fg-subtle">{label}</div>
        {icon && <span className="text-fg-subtle">{icon}</span>}
      </div>
      <div className={`num mt-1.5 font-display text-2xl font-semibold ${tone || 'text-fg'}`}>
        {value}
      </div>
      {hint && <div className="mt-1 text-xs leading-snug text-fg-subtle">{hint}</div>}
    </div>
  )
}

export function Modal({ title, description, onClose, children, footer, labelledBy = 'modal-title' }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ backgroundColor: 'rgb(var(--overlay) / 0.6)', backdropFilter: 'blur(3px)' }}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={labelledBy}
        className="card-raised w-full max-w-lg animate-scale-in"
      >
        <div className="flex items-start justify-between gap-4 border-b border-line px-6 py-4">
          <div>
            <h2 id={labelledBy} className="font-display text-lg font-semibold text-fg">
              {title}
            </h2>
            {description && <p className="mt-0.5 text-sm text-fg-muted">{description}</p>}
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close dialog"
            className="rounded-md p-1.5 text-fg-subtle transition hover:bg-sunken hover:text-fg"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 strokeWidth="1.5" strokeLinecap="round" aria-hidden="true" focusable="false">
              <path d="m6 6 12 12M18 6 6 18" />
            </svg>
          </button>
        </div>
        <div className="p-6">{children}</div>
        {footer && (
          <div className="flex justify-end gap-2 border-t border-line bg-sunken/50 px-6 py-4">
            {footer}
          </div>
        )}
      </div>
    </div>
  )
}
