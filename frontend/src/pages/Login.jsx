import { useState } from 'react'
import { useNavigate, useLocation, Link } from 'react-router-dom'
import { Logo } from '../components/Logo'
import { useSession } from '../components/session'
import { Banner } from '../components/ui'

const POINTS = [
  'Synthetic data generated under your own Apache Atlas classifications',
  'Differential privacy enforced automatically on sensitive columns',
  'Relational integrity preserved across related tables',
]

export default function Login() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const { signIn } = useSession()
  const navigate = useNavigate()
  const location = useLocation()
  const from = location.state?.from || '/app'

  function submit(event) {
    event.preventDefault()
    if (!email.trim()) {
      setError('Enter your OCBC email address.')
      return
    }
    setBusy(true)
    setError(null)
    signIn(email.trim())
    navigate(from, { replace: true })
  }

  return (
    <div className="grid min-h-[100dvh] lg:grid-cols-[1.05fr_1fr]">
      {/* Brand panel. Hidden below lg so the form owns the small viewport. */}
      <aside className="relative hidden overflow-hidden bg-[rgb(var(--brand-deep))] lg:flex lg:flex-col lg:justify-between lg:p-12">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 opacity-[0.16]"
          style={{
            backgroundImage:
              'radial-gradient(circle at 22% 18%, rgb(255 255 255 / 0.55) 0, transparent 42%), radial-gradient(circle at 78% 76%, rgb(255 255 255 / 0.30) 0, transparent 46%)',
          }}
        />
        <div className="relative">
          <img
            src="/brand/logo-full-dark.png"
            alt="OCBC DataCraft, powered by GDO"
            className="h-14 w-auto"
            draggable={false}
          />
        </div>

        <div className="relative max-w-lg">
          <h1 className="font-display text-4xl font-bold leading-[1.1] tracking-tight text-white">
            Production-grade data
            <br />
            without production risk.
          </h1>
          <p className="mt-5 text-base leading-relaxed text-white/75">
            Generate statistically faithful datasets for development, testing and
            analytics, governed by the classifications your data office already
            maintains.
          </p>

          <ul className="mt-8 space-y-3">
            {POINTS.map((point) => (
              <li key={point} className="flex gap-3 text-sm text-white/80">
                <svg
                  width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                  strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                  className="mt-0.5 shrink-0 text-white/50" aria-hidden="true" focusable="false"
                >
                  <path d="m5 12 5 5L20 7" />
                </svg>
                {point}
              </li>
            ))}
          </ul>
        </div>

        <p className="relative text-xs text-white/45">
          Internal proof of concept. Group Data Office.
        </p>
      </aside>

      {/* Form panel */}
      <main className="flex items-center justify-center px-6 py-12 sm:px-10">
        <div className="w-full max-w-[400px]">
          <div className="lg:hidden">
            <Logo height={40} />
          </div>

          <h2 className="mt-8 font-display text-2xl font-semibold tracking-tight text-fg lg:mt-0">
            Sign in to continue
          </h2>
          <p className="mt-1.5 text-sm text-fg-muted">
            Access your synthetic data workspace.
          </p>

          <Banner tone="warn" title="Demo sign-in">
            This screen does not authenticate anyone. The API has no auth and any
            email is accepted. Real access control is required before this
            platform touches production data.
          </Banner>

          <form onSubmit={submit} className="mt-2 space-y-4" noValidate>
            {error && <Banner tone="danger">{error}</Banner>}

            <div>
              <label className="label" htmlFor="login-email">
                Email address
              </label>
              <input
                id="login-email"
                type="email"
                autoComplete="username"
                className="input mt-1.5"
                placeholder="name@ocbc.com"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
              />
            </div>

            <div>
              <div className="flex items-baseline justify-between">
                <label className="label" htmlFor="login-password">
                  Password
                </label>
                <span className="text-xs text-fg-subtle">Not checked</span>
              </div>
              <input
                id="login-password"
                type="password"
                autoComplete="current-password"
                className="input mt-1.5"
                placeholder="••••••••"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
              />
            </div>

            <button type="submit" className="btn-primary w-full py-2.5" disabled={busy}>
              {busy ? 'Signing in…' : 'Sign in'}
            </button>
          </form>

          <p className="mt-6 text-center text-sm text-fg-muted">
            <Link to="/" className="font-medium text-accent-text hover:underline">
              Back to overview
            </Link>
          </p>
        </div>
      </main>
    </div>
  )
}
