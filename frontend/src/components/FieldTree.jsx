import { useState } from 'react'
import { ChevronIcon, CloseIcon, InfoIcon } from './icons'
import { Impact } from './ui'

/** Read/write a nested value by the dotted `path` the schema loader emits. */
export function getPath(object, path) {
  return path
    .split('.')
    .reduce((cursor, key) => (cursor == null ? undefined : cursor[key]), object)
}

export function setPath(object, path, value) {
  const parts = path.split('.')
  const next = structuredClone(object ?? {})
  let cursor = next
  parts.slice(0, -1).forEach((part) => {
    if (typeof cursor[part] !== 'object' || cursor[part] === null) cursor[part] = {}
    cursor = cursor[part]
  })
  cursor[parts[parts.length - 1]] = value
  return next
}

/** `conditions: [{field: './private', is: true}]` — '/x' is table-scoped,
 *  './x' is a sibling of the current node. */
function conditionsMet(node, values) {
  if (!node.conditions?.length) return true
  return node.conditions.every((condition) => {
    const reference = condition.field || ''
    const base = node.path.split('.').slice(0, -1)
    let target
    if (reference.startsWith('./')) {
      target = [...base, reference.slice(2)].join('.')
    } else if (reference.startsWith('/')) {
      // table-scoped: resolve against the nearest table container
      target = [...base, reference.slice(1)].join('.')
    } else {
      target = reference
    }
    return getPath(values, target) === condition.is
  })
}

function Docs({ text }) {
  const [open, setOpen] = useState(false)
  if (!text) return null
  return (
    <span className="relative">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        aria-label="Show documentation"
        className="ml-1 inline-flex items-center rounded p-0.5 text-fg-subtle hover:text-accent-text"
      >
        <InfoIcon size={14} />
      </button>
      {open && (
        <div className="absolute left-0 top-6 z-30 w-96 rounded-md border border-line bg-surface p-3 text-xs leading-relaxed text-fg-muted shadow-lg">
          {text}
        </div>
      )}
    </span>
  )
}

function MultiSelect({ options, value, onChange, placeholder }) {
  const selected = Array.isArray(value) ? value : value ? [value] : []
  return (
    <div className="rounded-md border border-line-strong px-2 py-1.5">
      <div className="flex flex-wrap gap-1">
        {selected.map((item) => (
          <span
            key={item}
            className="pill flex items-center gap-1 bg-sunken text-fg-muted"
          >
            {item}
            <button
              type="button"
              aria-label={`Remove ${item}`}
              className="inline-flex items-center rounded p-0.5 text-fg-subtle hover:text-danger-text"
              onClick={() => onChange(selected.filter((v) => v !== item))}
            >
              <CloseIcon size={12} />
            </button>
          </span>
        ))}
        <select
          className="min-w-[120px] flex-1 border-0 bg-transparent text-sm outline-none"
          value=""
          onChange={(event) => {
            if (event.target.value) onChange([...selected, event.target.value])
          }}
        >
          <option value="">{placeholder || 'add…'}</option>
          {options
            .filter((option) => !selected.includes(option.value))
            .map((option) => (
              <option key={String(option.value)} value={option.value}>
                {option.label}
              </option>
            ))}
        </select>
      </div>
    </div>
  )
}

function ObjectsArray({ node, values, onChange, allTables }) {
  const items = getPath(values, node.path) || []

  function update(index, field, value) {
    const next = structuredClone(items)
    next[index] = { ...next[index], [field]: value }
    onChange(setPath(values, node.path, next))
  }

  return (
    <div className="space-y-3">
      {items.map((item, index) => (
        <div key={index} className="rounded-md border border-line p-3">
          <div className="mb-2 flex justify-end">
            <button
              type="button"
              className="btn-link text-danger-text hover:bg-danger-subtle"
              onClick={() =>
                onChange(setPath(values, node.path, items.filter((_, i) => i !== index)))
              }
            >
              <CloseIcon size={12} />
              Remove
            </button>
          </div>
          {node.itemSchema?.map((field) => {
            // parent_column_names depends on the chosen parent table
            let options = field.options || []
            if (field.optionSource === 'dynamic-table-columns') {
              const parent = item.parent_table_name
              options = (allTables?.[parent] || []).map((column) => ({
                label: column,
                value: column,
              }))
            }
            const value = item[field.key]
            return (
              <div key={field.key} className="mb-2">
                <div className="mb-1 flex items-center text-sm text-fg-muted">
                  {field.label}
                  <Docs text={field.documentation} />
                  <span className="ml-2">
                    <Impact tags={field.tags} />
                  </span>
                </div>
                {field.multi ? (
                  <MultiSelect
                    options={options}
                    value={value}
                    onChange={(next) => update(index, field.key, next)}
                  />
                ) : (
                  <select
                    className="input"
                    value={value ?? ''}
                    onChange={(event) => update(index, field.key, event.target.value)}
                  >
                    <option value="">— select —</option>
                    {options.map((option) => (
                      <option key={String(option.value)} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                )}
              </div>
            )
          })}
        </div>
      ))}
      <button
        type="button"
        className="w-full rounded-md border border-dashed border-line-strong py-2 min-h-[40px] cursor-pointer text-sm text-fg-muted hover:border-accent hover:text-accent-text"
        onClick={() => onChange(setPath(values, node.path, [...items, {}]))}
      >
        Add +
      </button>
    </div>
  )
}

export function FieldNode({ node, values, onChange, allTables, depth = 0 }) {
  const [open, setOpen] = useState(depth < 2)
  if (!conditionsMet(node, values)) return null

  const value = getPath(values, node.path)
  const indent = { marginLeft: depth ? 14 : 0 }

  // container nodes
  if (node.children?.length) {
    return (
      <div style={indent} className="mb-2">
        <button
          type="button"
          onClick={() => setOpen((current) => !current)}
          className="flex w-full items-center gap-2 rounded py-1.5 text-left"
          aria-expanded={open}
        >
          <ChevronIcon size={14} open={open} className="shrink-0 text-fg-subtle" />
          <span className="text-sm font-medium text-fg">{node.label}</span>
          <Impact tags={node.tags} />
          <Docs text={node.documentation} />
        </button>
        {open && (
          <div className="border-l border-line pl-3">
            {node.children.map((child) => (
              <FieldNode
                key={child.path}
                node={child}
                values={values}
                onChange={onChange}
                allTables={allTables}
                depth={depth + 1}
              />
            ))}
          </div>
        )}
      </div>
    )
  }

  const header = (
    <div className="mb-1 flex items-center gap-2">
      <span className="text-sm text-fg-muted">{node.label}</span>
      <Docs text={node.documentation} />
      <Impact tags={node.tags} />
      {node.required && <span className="text-xs text-danger">required</span>}
    </div>
  )

  if (node.type === 'objects-array') {
    return (
      <div style={indent} className="mb-3">
        {header}
        <ObjectsArray node={node} values={values} onChange={onChange} allTables={allTables} />
      </div>
    )
  }

  if (node.type === 'boolean') {
    return (
      <div style={indent} className="mb-2 flex items-center gap-2">
        <input
          type="checkbox"
          className="h-4 w-4 rounded border-line-strong text-accent-text"
          checked={Boolean(value)}
          onChange={(event) => onChange(setPath(values, node.path, event.target.checked))}
        />
        <span className="text-sm text-fg-muted">{node.label}</span>
        <Docs text={node.documentation} />
        <Impact tags={node.tags} />
      </div>
    )
  }

  if (node.type === 'select') {
    return (
      <div style={indent} className="mb-3">
        {header}
        {node.multi ? (
          <MultiSelect
            options={node.options || []}
            value={value}
            onChange={(next) => onChange(setPath(values, node.path, next))}
          />
        ) : (
          <select
            className="input"
            value={value ?? ''}
            onChange={(event) => onChange(setPath(values, node.path, event.target.value || null))}
          >
            <option value="">— select —</option>
            {(node.options || []).map((option) => (
              <option key={String(option.value)} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        )}
      </div>
    )
  }

  if (node.type === 'number') {
    return (
      <div style={indent} className="mb-3">
        {header}
        <input
          type="number"
          className="input"
          value={value ?? node.initialValue ?? ''}
          min={node.min}
          max={node.max}
          onChange={(event) =>
            onChange(
              setPath(values, node.path, event.target.value === '' ? null : Number(event.target.value)),
            )
          }
        />
      </div>
    )
  }

  return (
    <div style={indent} className="mb-3">
      {header}
      <input
        className="input"
        value={value ?? node.initialValue ?? ''}
        onChange={(event) => onChange(setPath(values, node.path, event.target.value))}
      />
    </div>
  )
}

export default function FieldTree({ sections, values, onChange, allTables, filter }) {
  const [query, setQuery] = useState('')

  function visible(node) {
    if (filter === 'high' && !node.highImpact) {
      const anyChild = (node.children || []).some((child) => visible(child))
      if (!anyChild) return false
    }
    if (query) {
      const match =
        node.label?.toLowerCase().includes(query.toLowerCase()) ||
        node.key?.toLowerCase().includes(query.toLowerCase())
      const anyChild = (node.children || []).some((child) => visible(child))
      if (!match && !anyChild) return false
    }
    return true
  }

  return (
    <div>
      <input
        className="input mb-4"
        placeholder="Search Parameters"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
      />
      {sections.map((section) => {
        const fields = section.fields.filter(visible)
        if (!fields.length) return null
        return (
          <div key={section.name} className="mb-6">
            <h3 className="text-sm font-semibold text-fg">{section.name}</h3>
            {section.documentation && (
              <p className="mb-3 mt-0.5 text-xs text-fg-subtle">{section.documentation}</p>
            )}
            <div className="mt-2">
              {fields.map((node) => (
                <FieldNode
                  key={node.path}
                  node={node}
                  values={values}
                  onChange={onChange}
                  allTables={allTables}
                />
              ))}
            </div>
          </div>
        )
      })}
    </div>
  )
}
