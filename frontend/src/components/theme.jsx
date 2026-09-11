import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'

const STORAGE_KEY = 'ocbc-datacraft-theme'
const ThemeContext = createContext({ theme: 'light', preference: 'system', setPreference: () => {} })

function systemTheme() {
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

/**
 * Three states, not two: light, dark, and "follow the system".
 *
 * Defaulting to the OS setting means the app matches the rest of the desktop on
 * first load; once the user picks explicitly, that choice wins and persists.
 * A two-state toggle silently overrides the OS preference on first paint.
 */
export function ThemeProvider({ children }) {
  const [preference, setPreferenceState] = useState(
    () => localStorage.getItem(STORAGE_KEY) || 'system',
  )
  const [resolved, setResolved] = useState(() =>
    (localStorage.getItem(STORAGE_KEY) || 'system') === 'system'
      ? systemTheme()
      : localStorage.getItem(STORAGE_KEY),
  )

  useEffect(() => {
    const theme = preference === 'system' ? systemTheme() : preference
    setResolved(theme)
    document.documentElement.setAttribute('data-theme', theme)
  }, [preference])

  // Keep following the OS while the preference is "system".
  useEffect(() => {
    if (preference !== 'system') return undefined
    const query = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = () => {
      const theme = systemTheme()
      setResolved(theme)
      document.documentElement.setAttribute('data-theme', theme)
    }
    query.addEventListener('change', onChange)
    return () => query.removeEventListener('change', onChange)
  }, [preference])

  const setPreference = useCallback((next) => {
    setPreferenceState(next)
    if (next === 'system') localStorage.removeItem(STORAGE_KEY)
    else localStorage.setItem(STORAGE_KEY, next)
  }, [])

  const value = useMemo(
    () => ({ theme: resolved, preference, setPreference }),
    [resolved, preference, setPreference],
  )

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export const useTheme = () => useContext(ThemeContext)

const OPTIONS = [
  {
    value: 'light',
    label: 'Light',
    icon: (
      <>
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
      </>
    ),
  },
  {
    value: 'system',
    label: 'System',
    icon: (
      <>
        <rect x="3" y="4" width="18" height="12" rx="2" />
        <path d="M8 20h8M12 16v4" />
      </>
    ),
  },
  {
    value: 'dark',
    label: 'Dark',
    icon: <path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z" />,
  },
]

/** Segmented control rather than a single toggle, so the current mode is
 *  readable at a glance instead of inferred from an icon. */
export function ThemeToggle() {
  const { preference, setPreference } = useTheme()
  return (
    <div
      role="radiogroup"
      aria-label="Colour theme"
      className="inline-flex items-center gap-0.5 rounded-lg border border-line bg-sunken p-0.5"
    >
      {OPTIONS.map((option) => {
        const active = preference === option.value
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={active}
            aria-label={option.label}
            title={option.label}
            onClick={() => setPreference(option.value)}
            className={`flex h-7 w-7 items-center justify-center rounded-md transition ${
              active
                ? 'bg-surface text-accent-text shadow-sm'
                : 'text-fg-subtle hover:text-fg-muted'
            }`}
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.75"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
              focusable="false"
            >
              {option.icon}
            </svg>
          </button>
        )
      })}
    </div>
  )
}
