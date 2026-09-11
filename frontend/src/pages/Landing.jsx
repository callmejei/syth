import { Link } from 'react-router-dom'
import { Logo } from '../components/Logo'
import { ThemeToggle } from '../components/theme'

/* Every figure below is measured, not aspirational. Sources:
   - fidelity / detection / utility: scripts/compare_engines.py on the 4,000-row
     customer sample, ARF engine, no differential privacy
   - FK integrity and PII checks: scripts/smoke_test.py, 24/24
   - schema coverage: scripts/schema_coverage.py                                */
const HEADLINE_METRICS = [
  { value: '98.6%', label: 'Statistical fidelity', note: 'per-column distribution match' },
  { value: '0.52', label: 'Detection AUC', note: '0.50 means indistinguishable from real' },
  { value: '95%', label: 'Utility retained', note: 'model trained on synthetic vs on real' },
  { value: '0', label: 'Real PII reproduced', note: 'across name, email, phone, NRIC' },
]

const CAPABILITIES = [
  {
    title: 'Governed by your catalog',
    body: 'Column treatment is decided by Apache Atlas classifications, not inferred from the data. A column tagged PII is pseudonymised because the catalog says so, and the report records that reason.',
    points: ['PII pseudonymised', 'PCI suppressed entirely', 'Quasi-identifiers generalised'],
  },
  {
    title: 'Privacy that is not optional',
    body: 'When Atlas marks a column SENSITIVE, differential privacy switches on for that run whether or not anyone ticks the box. The epsilon budget covers every data-dependent step of training.',
    points: ['Pure epsilon-DP, delta = 0', 'Budget shown per stage', 'Refused, never faked'],
  },
  {
    title: 'Relationships preserved',
    body: 'Related tables stay coherent. Every foreign key resolves, cardinality per parent is kept, and a customer segment still drives the size of that customer’s transactions.',
    points: ['Zero orphan keys', 'Cardinality preserved', 'Cross-table correlation held'],
  },
]

const ENGINES = [
  {
    name: 'SPN',
    full: 'Sum-Product Network',
    dp: true,
    best: 'Sensitive data and related tables',
    note: 'Differential privacy supported, because every learned parameter is a count.',
  },
  {
    name: 'ARF',
    full: 'Adversarial Random Forest',
    dp: false,
    best: 'Single wide tables',
    note: 'Trains until a classifier can no longer separate synthetic from real. No DP.',
  },
]

function Section({ id, children, className = '' }) {
  return (
    <section id={id} className={`mx-auto w-full max-w-[1180px] px-6 lg:px-10 ${className}`}>
      {children}
    </section>
  )
}

export default function Landing() {
  return (
    <div className="min-h-[100dvh] bg-canvas">
      {/* Navigation: single line, 68px */}
      <header className="sticky top-0 z-40 border-b border-line bg-canvas/85 backdrop-blur">
        <div className="mx-auto flex h-[68px] w-full max-w-[1180px] items-center justify-between px-6 lg:px-10">
          <Logo height={44} />
          <nav aria-label="Primary" className="flex items-center gap-1 sm:gap-3">
            <a
              href="#how"
              className="hidden rounded-md px-3 py-2 text-sm font-medium text-fg-muted transition hover:text-fg sm:block"
            >
              How it works
            </a>
            <a
              href="#engines"
              className="hidden rounded-md px-3 py-2 text-sm font-medium text-fg-muted transition hover:text-fg sm:block"
            >
              Engines
            </a>
            <ThemeToggle />
            <Link to="/login" className="btn-primary ml-1">
              Sign in
            </Link>
          </nav>
        </div>
      </header>

      {/* Hero: asymmetric split, 2-line headline, 20-word subtext, one CTA pair */}
      <Section className="pt-16 lg:pt-24">
        <div className="grid items-center gap-12 lg:grid-cols-[1.15fr_1fr]">
          <div>
            <h1 className="font-display text-[2.1rem] font-bold leading-[1.08] tracking-tight text-fg sm:text-[2.5rem] lg:text-[2.9rem]">
              Synthetic data your
              <br />
              <span className="text-accent-text">data office</span> can sign off.
            </h1>
            <p className="mt-6 max-w-[52ch] text-lg leading-relaxed text-fg-muted">
              Generate realistic datasets for development and analytics, with every
              column treated according to its Apache Atlas classification.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Link to="/login" className="btn-primary px-5 py-2.5 text-base">
                Open the workspace
              </Link>
              <a href="#how" className="btn-ghost px-5 py-2.5 text-base">
                See how it works
              </a>
            </div>
          </div>

          <div className="relative">
            <div
              aria-hidden="true"
              className="absolute -inset-6 rounded-[2rem] opacity-70"
              style={{
                background:
                  'radial-gradient(60% 60% at 70% 30%, rgb(var(--accent) / 0.16) 0, transparent 70%)',
              }}
            />
            <div className="card-raised relative p-6">
              <div className="text-2xs font-semibold uppercase tracking-wider text-fg-subtle">
                Latest evaluation
              </div>
              <div className="mt-4 grid grid-cols-2 gap-5">
                {HEADLINE_METRICS.map((metric) => (
                  <div key={metric.label}>
                    <div className="num font-display text-3xl font-bold text-fg">
                      {metric.value}
                    </div>
                    <div className="mt-1 text-sm font-medium text-fg">{metric.label}</div>
                    <div className="mt-0.5 text-xs leading-snug text-fg-subtle">
                      {metric.note}
                    </div>
                  </div>
                ))}
              </div>
              <p className="mt-6 border-t border-line pt-4 text-xs leading-relaxed text-fg-subtle">
                Measured on a 4,000-row customer and orders dataset. Reproduce with{' '}
                <code className="font-mono text-[11px] text-fg-muted">compare_engines.py</code>.
              </p>
            </div>
          </div>
        </div>
      </Section>

      {/* Capability trio: stacked rows, not three equal cards */}
      <Section id="how" className="pt-24 lg:pt-32">
        <h2 className="max-w-[20ch] font-display text-3xl font-bold tracking-tight text-fg lg:text-4xl">
          Governance decides, the model follows.
        </h2>
        <div className="mt-12 divide-y divide-line border-y border-line">
          {CAPABILITIES.map((capability, index) => (
            <div key={capability.title} className="grid gap-6 py-8 lg:grid-cols-[auto_1fr_auto] lg:gap-12">
              <div className="num font-display text-2xl font-bold text-accent-text lg:w-16">
                {String(index + 1).padStart(2, '0')}
              </div>
              <div className="max-w-[62ch]">
                <h3 className="font-display text-xl font-semibold text-fg">
                  {capability.title}
                </h3>
                <p className="mt-2 leading-relaxed text-fg-muted">{capability.body}</p>
              </div>
              <ul className="space-y-2 lg:w-64">
                {capability.points.map((point) => (
                  <li
                    key={point}
                    className="flex items-start gap-2 text-sm text-fg-muted"
                  >
                    <span
                      aria-hidden="true"
                      className="mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full bg-accent"
                    />
                    {point}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </Section>

      {/* Engines: two-up comparison */}
      <Section id="engines" className="pt-24 lg:pt-32">
        <h2 className="max-w-[24ch] font-display text-3xl font-bold tracking-tight text-fg lg:text-4xl">
          Two engines, chosen per dataset.
        </h2>
        <p className="mt-4 max-w-[62ch] leading-relaxed text-fg-muted">
          Both are implemented in house on open licences, so nothing here carries a
          per-seat cost or a production licence restriction.
        </p>

        <div className="mt-10 grid gap-5 md:grid-cols-2">
          {ENGINES.map((engine) => (
            <article key={engine.name} className="card flex flex-col p-6">
              <div className="flex items-baseline justify-between gap-3">
                <h3 className="font-display text-2xl font-bold text-fg">{engine.name}</h3>
                <span
                  className={`pill ${
                    engine.dp
                      ? 'bg-success-subtle text-success-text'
                      : 'bg-warning-subtle text-warning-text'
                  }`}
                >
                  {engine.dp ? 'Differential privacy' : 'No differential privacy'}
                </span>
              </div>
              <div className="mt-1 text-sm text-fg-subtle">{engine.full}</div>
              <p className="mt-4 flex-1 leading-relaxed text-fg-muted">{engine.note}</p>
              <div className="mt-5 border-t border-line pt-4">
                <div className="text-2xs font-semibold uppercase tracking-wider text-fg-subtle">
                  Best for
                </div>
                <div className="mt-1 font-medium text-fg">{engine.best}</div>
              </div>
            </article>
          ))}
        </div>
      </Section>

      {/* Honest limitations. This is what makes the numbers above credible. */}
      <Section className="pt-24 lg:pt-32">
        <div className="card bg-sunken p-8 lg:p-10">
          <h2 className="font-display text-2xl font-bold tracking-tight text-fg">
            What this proof of concept does not do yet
          </h2>
          <div className="mt-6 grid gap-x-12 gap-y-4 text-sm leading-relaxed text-fg-muted md:grid-cols-2">
            <p>
              <span className="font-semibold text-fg">No authentication.</span> The API
              is open and the sign-in screen is a demo gate. Real access control is
              required before production data.
            </p>
            <p>
              <span className="font-semibold text-fg">Differential privacy costs accuracy.</span>{' '}
              On small tables the loss is severe. It needs volume to be useful.
            </p>
            <p>
              <span className="font-semibold text-fg">Time series is shallow.</span> Rows
              are ordered within an entity, but there is no sequence model.
            </p>
            <p>
              <span className="font-semibold text-fg">Not load tested.</span> The largest
              verified run is roughly 24,000 rows across four related tables.
            </p>
          </div>
        </div>
      </Section>

      <Section className="py-24 lg:py-32">
        <div className="flex flex-col items-start justify-between gap-6 border-t border-line pt-10 sm:flex-row sm:items-center">
          <div>
            <h2 className="font-display text-2xl font-bold tracking-tight text-fg">
              Ready to try it on your own extract?
            </h2>
            <p className="mt-2 text-fg-muted">
              Upload a table, tag it in Atlas, and review the report.
            </p>
          </div>
          <Link to="/login" className="btn-primary shrink-0 px-5 py-2.5 text-base">
            Open the workspace
          </Link>
        </div>
      </Section>

      <footer className="border-t border-line">
        <div className="mx-auto flex w-full max-w-[1180px] flex-col gap-4 px-6 py-8 sm:flex-row sm:items-center sm:justify-between lg:px-10">
          <Logo height={28} />
          <p className="text-xs text-fg-subtle">
            Internal proof of concept. Group Data Office. Not a production system.
          </p>
        </div>
      </footer>
    </div>
  )
}
