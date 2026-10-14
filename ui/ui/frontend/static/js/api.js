const streams = new Map();

async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      ...(options.body ? { 'Content-Type': 'application/json' } : {}),
      ...(options.headers || {}),
    },
  });
  const contentType = response.headers.get('content-type') || '';
  let data = null;
  if (contentType.includes('application/json')) {
    data = await response.json().catch(() => ({}));
  } else if (!response.ok) {
    data = { message: await response.text().catch(() => '') };
  }
  if (!response.ok) {
    const message = data?.message || data?.error || data?.detail || `Request failed (${response.status})`;
    throw new Error(message);
  }
  return data;
}

const get = path => request(path);
const post = (path, payload = {}) => request(path, { method: 'POST', body: JSON.stringify(payload) });

export const api = {
  status: () => get('/api/status'),
  policies: () => get('/api/policies'),
  datasets: () => get('/api/datasets'),
  checkpoints: () => get('/api/checkpoints'),
  prepareDataset: (datasetId, purpose) => post('/api/datasets/prepare', { dataset_id: datasetId, purpose }),
  datasetPreparation: jobId => get(`/api/datasets/prepare/${encodeURIComponent(jobId)}`),
  deleteDataset: path => post('/api/datasets/delete', { path }),
  trainStart: payload => post('/api/train/start', payload),
  trainStop: () => post('/api/train/stop'),
  trainReset: () => post('/api/train/reset'),
  trainResume: checkpointPath => post('/api/train/resume', checkpointPath ? { checkpoint_path: checkpointPath } : {}),
  trainStatus: () => get('/api/train/status'),
  checkpointDelete: path => post('/api/checkpoints/delete', { path }),
  inferenceMetadata: (checkpoint, actionHorizon, controlHz) => {
    const params = new URLSearchParams({ checkpoint });
    if (actionHorizon) params.set('action_horizon', actionHorizon);
    if (controlHz) params.set('control_hz', controlHz);
    return get(`/api/inference/checkpoint_metadata?${params}`);
  },
  serverStart: payload => post('/api/inference/start_server', payload),
  serverStop: () => post('/api/inference/stop_server'),
  bridgeStart: payload => post('/api/inference/start_bridge', payload),
  bridgeStop: () => post('/api/inference/stop_bridge'),
  viewer: datasetId => get(`/api/datasets/${encodeURIComponent(datasetId)}/viewer`),
  episodes: (datasetId, { offset = 0, limit = 300, task = '', signal } = {}) => {
    const params = new URLSearchParams({ offset: String(offset), limit: String(limit) });
    if (task) params.set('task', task);
    return request(`/api/datasets/${encodeURIComponent(datasetId)}/episodes?${params}`, { signal });
  },
  actions: (datasetId, episode, options = {}) => request(`/api/datasets/${encodeURIComponent(datasetId)}/actions?episode=${encodeURIComponent(episode)}`, options),
  datasetInfoForEvaluation: payload => post('/api/model_validation/dataset_info', payload),
  startEvaluation: payload => post('/api/model_validation/evaluate', payload),
  activeEvaluation: () => get('/api/model_validation/jobs/active'),
  evaluationStatus: jobId => get(`/api/model_validation/job/${encodeURIComponent(jobId)}`),
  stopEvaluation: jobId => post(`/api/model_validation/job/${encodeURIComponent(jobId)}/stop`),
};

export function subscribeLogs(name, endpoint, onLine, onState = () => {}) {
  const current = streams.get(name);
  if (current) return current;
  const source = new EventSource(endpoint);
  streams.set(name, source);
  source.onopen = () => onState('connected');
  source.onmessage = event => onLine(event.data);
  source.onerror = () => onState(source.readyState === EventSource.CLOSED ? 'disconnected' : 'reconnecting');
  return source;
}

export function closeLogStream(name) {
  const source = streams.get(name);
  if (source) source.close();
  streams.delete(name);
}
