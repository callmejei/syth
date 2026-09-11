import { Link } from 'react-router-dom'
import { PageHeader } from '../components/ui'

const CARDS = [
  {
    to: '/app/admin/model-types',
    title: 'Model types',
    body: 'Enable or disable generative engines, and see their licence position.',
  },
  {
    to: '/app/admin/compute',
    title: 'Compute settings',
    body: 'Manage compute profiles and per-job limits.',
  },
  {
    to: '/app/admin/audit-logs',
    title: 'Audit Logs',
    body: 'Every configuration and policy change, with actor and timestamp.',
  },
  {
    to: '/app/governance',
    title: 'Data protection policy',
    body: 'Apache Atlas classifications and the strategy applied to each column.',
  },
]

export default function Admin() {
  return (
    <div>
      <PageHeader title="Admin Settings" />
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {CARDS.map((card) => (
          <Link
            key={card.to}
            to={card.to}
            className="card p-5 transition hover:border-accent/40 hover:shadow"
          >
            <div className="font-medium text-fg">{card.title}</div>
            <p className="mt-1 text-sm text-fg-subtle">{card.body}</p>
          </Link>
        ))}
      </div>
    </div>
  )
}
