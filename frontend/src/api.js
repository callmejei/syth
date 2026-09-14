const BASE = '/api'

/**
 * Trigger a browser download for a GET endpoint that returns a file.
 *
 * The response is pulled as a blob rather than pointed at with window.open so
 * that an error response renders as a message instead of navigating the tab to
 * a page of raw JSON.
 */
async function download(path) {
  const response = await fetch(`${BASE}${path}`)
  if (!response.ok) {
    let detail
    try {
      detail = (await response.json()).detail
    } catch {
      detail = await response.text()
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }

  const disposition = response.headers.get('Content-Disposition') || ''
  const match = disposition.match(/filename="?([^"]+)"?/)
  const blob = await response.blob()
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = match ? match[1] : 'download'
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  // Revoking immediately can cancel the download in some browsers.
  setTimeout(() => URL.revokeObjectURL(url), 10_000)
}

async function request(path, options = {}) {
  const response = await fetch(`${BASE}${path}`, {
    headers: options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!response.ok) {
    let detail
    try {
      detail = (await response.json()).detail
    } catch {
      detail = await response.text()
    }
    const error = new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
    // Callers need the status to tell a refusal they can offer a way around
    // (409: the catalog does not cover these tables) from a plain failure.
    error.status = response.status
    error.detail = detail
    throw error
  }
  if (response.status === 204) return null
  return response.json()
}

export const api = {
  // schema
  schemaTree: (dataSourceId, tables) =>
    request(
      `/schema/tree?data_source_id=${dataSourceId || ''}&tables=${(tables || []).join(',')}`,
    ),
  schemaGroups: () => request('/schema/groups'),

  // data sources
  dataSources: () => request('/data-sources'),
  dataSource: (id) => request(`/data-sources/${id}`),
  uploadDataSource: (name, files, projectId) => {
    const form = new FormData()
    form.append('name', name)
    files.forEach((file) => form.append('files', file))
    if (projectId) form.append('project_id', projectId)
    return request('/data-sources/upload', { method: 'POST', body: form })
  },
  downloadDataSource: (id, { table, fmt = 'csv' } = {}) =>
    download(
      `/data-sources/${id}/download?fmt=${fmt}${table ? `&table=${encodeURIComponent(table)}` : ''}`,
    ),
  dataSourceUsage: (id) => request(`/data-sources/${id}/usage`),
  deleteDataSource: (id, cascade = false) =>
    request(`/data-sources/${id}?cascade=${cascade}`, { method: 'DELETE' }),
  refreshAtlas: (id) => request(`/data-sources/${id}/refresh-atlas`, { method: 'POST' }),
  policy: (id, mode = 'balanced') => request(`/data-sources/${id}/policy?mode=${mode}`),

  // configurations
  configurations: () => request('/configurations'),
  configuration: (id) => request(`/configurations/${id}`),
  createConfiguration: (body) =>
    request('/configurations', { method: 'POST', body: JSON.stringify(body) }),
  updateConfiguration: (id, body) =>
    request(`/configurations/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),
  // requireCatalog=false derives the schema from the data alone, for tables the
  // catalog does not cover. The response flags that it was not catalog-backed.
  autoconfigure: (id, { requireCatalog = true } = {}) =>
    request(`/configurations/${id}/autoconfigure?require_catalog=${requireCatalog}`, {
      method: 'POST',
    }),
  train: (id, body) =>
    request(`/configurations/${id}/train`, { method: 'POST', body: JSON.stringify(body) }),

  // jobs & models
  jobs: () => request('/jobs'),
  job: (id) => request(`/jobs/${id}`),
  preview: (jobId, table, limit = 20) =>
    request(`/jobs/${jobId}/preview?table=${table}&limit=${limit}`),
  downloadOutput: (jobId, { table, fmt = 'csv' } = {}) =>
    download(
      `/jobs/${jobId}/download?fmt=${fmt}${table ? `&table=${encodeURIComponent(table)}` : ''}`,
    ),
  generate: (body) => request('/generate', { method: 'POST', body: JSON.stringify(body) }),
  models: () => request('/models'),
  model: (id) => request(`/models/${id}`),
  report: (id) => request(`/models/${id}/report`),

  // admin
  modelTypes: () => request('/admin/model-types'),
  updateModelType: (id, body) =>
    request(`/admin/model-types/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),
  compute: () => request('/admin/compute'),
  updateProfile: (id, body) =>
    request(`/admin/compute/profiles/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),
  setting: (key) => request(`/admin/settings/${key}`),
  putSetting: (key, value) =>
    request(`/admin/settings/${key}`, { method: 'PUT', body: JSON.stringify(value) }),
  auditLogs: () => request('/admin/audit-logs'),
  atlasStatus: () => request('/admin/atlas/status'),

  // projects
  projects: () => request('/projects'),
  project: (id) => request(`/projects/${id}`),
  createProject: (body) => request('/projects', { method: 'POST', body: JSON.stringify(body) }),
}

export default api
