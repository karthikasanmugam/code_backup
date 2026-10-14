import { api, closeLogStream, subscribeLogs } from './api.js';
import { state } from './state.js';

const $ = id => document.getElementById(id);
const lossPoints = [];
let lossDatasetLabel = '';
let sampleMetadata = null;
let sampleEpisodes = [];
let sampleToken = 0;
let sampleSelected = null;
let pendingResumeRun = null;
let pendingResumePolicy = '';

function notice(message, kind = 'info') {
  window.dispatchEvent(new CustomEvent('app:notice', { detail: { message, kind } }));
}

function create(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderPolicies() {
  const root = $('model-selector');
  root.replaceChildren();
  const entries = Object.entries(state.policies || {});
  if (!entries.length) {
    root.append(create('p', 'muted', 'No policies are available from the backend.'));
    return;
  }
  if (!state.selectedPolicy || !state.policies[state.selectedPolicy]) state.selectedPolicy = entries[0][0];
  entries.forEach(([key, policy]) => {
    const label = create('label', `model-option model-${key}${state.selectedPolicy === key ? ' selected' : ''}`);
    const radio = document.createElement('input');
    radio.type = 'radio'; radio.name = 'training-policy'; radio.value = key;
    radio.checked = state.selectedPolicy === key;
    radio.addEventListener('change', () => {
      state.selectedPolicy = key;
      try { window.localStorage.setItem('daksha.selectedPolicy', key); } catch (_) { /* selection remains in memory */ }
      renderPolicies();
      renderHyperparameters();
      updateStartAvailability();
    });
    const copy = create('span', 'model-copy');
    copy.append(create('strong', '', policy.name || key));
    label.append(radio, copy);
    root.append(label);
  });
  renderHyperparameters();
}

function renderHyperparameters() {
  const root = $('hyperparameters');
  root.replaceChildren();
  const schema = state.policies[state.selectedPolicy]?.hyperparameters || {};
  const entries = Object.entries(schema);
  if (!entries.length) {
    root.append(create('p', 'muted', 'This policy does not expose hyperparameter controls.'));
    return;
  }
  entries.forEach(([name, spec]) => {
    const field = create('div', spec.type === 'boolean' || spec.type === 'bool' ? 'field checkbox-field' : 'field');
    const label = create('label', '', spec.label || name.replaceAll('_', ' '));
    label.htmlFor = `hp-${name}`;
    const input = document.createElement('input');
    input.id = `hp-${name}`; input.name = name;
    const type = String(spec.type || 'number').toLowerCase();
    if (type === 'boolean' || type === 'bool') {
      input.type = 'checkbox'; input.checked = Boolean(spec.default);
      field.append(input, label);
    } else {
      input.type = type === 'string' || type === 'text' ? 'text' : 'number';
      input.value = spec.default ?? '';
      if (input.type === 'number') {
        if (spec.min !== undefined) input.min = spec.min;
        if (spec.max !== undefined) input.max = spec.max;
        input.step = spec.step ?? (Number.isInteger(Number(spec.default)) ? '1' : 'any');
      }
      field.append(label, input);
    }
    root.append(field);
  });
}

function applyHyperparameterValues(values) {
  // Reflect a resumed run's original parameters (from its train_config.json) into the
  // Training-configuration inputs. Only fields that exist in the current policy's schema
  // are touched; anything missing keeps its rendered default.
  if (!values || typeof values !== 'object') return;
  const schema = state.policies[state.selectedPolicy]?.hyperparameters || {};
  Object.entries(values).forEach(([name, value]) => {
    const input = $(`hp-${name}`);
    if (!input || !(name in schema) || value === null || value === undefined) return;
    const type = String(schema[name].type || 'number').toLowerCase();
    if (type === 'boolean' || type === 'bool') input.checked = Boolean(value);
    else input.value = value;
  });
}

function hyperparameterValues() {
  const result = {};
  const schema = state.policies[state.selectedPolicy]?.hyperparameters || {};
  Object.entries(schema).forEach(([name, spec]) => {
    const input = $(`hp-${name}`);
    if (!input) return;
    const type = String(spec.type || 'number').toLowerCase();
    if (type === 'boolean' || type === 'bool') result[name] = input.checked;
    else if (type === 'string' || type === 'text') result[name] = input.value;
    else result[name] = input.value === '' ? null : Number(input.value);
  });
  return result;
}

function updateStartAvailability() {
  $('train-start-btn').disabled = !state.selectedPolicy || !$('train-dataset').value || Boolean(state.training?.active) || startBusy || resumeBusy;
}

function updateTrainingDatasetOptions(datasets) {
  const select = $('train-dataset');
  const resumeDataset = pendingResumeRun ? datasetForRun(pendingResumeRun, datasets) : null;
  const prior = resumeDataset?.dataset_id || state.selectedDatasetId || select.value;
  select.replaceChildren(new Option(datasets.length ? 'Select a dataset' : 'No datasets found', ''));
  datasets.forEach(dataset => select.add(new Option(dataset.name || dataset.dataset_id, dataset.dataset_id)));
  if (datasets.some(item => item.dataset_id === prior)) select.value = prior;
  else if (datasets.length === 1 && !pendingResumeRun) select.value = datasets[0].dataset_id;
  state.selectedDatasetId = select.value;
  if (pendingResumeRun && datasets.length) {
    if (!resumeDataset) notice(`The checkpoint dataset ${pendingResumeRun.dataset_root || pendingResumeRun.task || ''} is not in the current dataset catalog.`, 'error');
    pendingResumeRun = null;
  }
  if (prior !== select.value && !state.training?.active) resetLossMetrics(select.value);
  renderSelectedDatasetInfo();
  updateStartAvailability();
  refreshTrainingSample(select.value);
}

function renderSelectedDatasetInfo() {
  const dataset = state.datasets.find(item => item.dataset_id === $('train-dataset').value);
  if (!dataset) {
    $('train-dataset-info').textContent = 'Choose a dataset from the backend catalog.';
    return;
  }
  const size = Number.isFinite(Number(dataset.size_gb)) ? `${dataset.size_gb} GB` : 'Size unavailable';
  $('train-dataset-info').textContent = `${dataset.dataset_id} · ${size} · ${dataset.created || 'Creation date unavailable'}`;
}

function metricFromLog(line) {
  const stepMatch = line.match(/\b(?:step|global[_ ]step|iteration|iter)\s*[:=# ]\s*([\d,]+(?:\.\d+)?)\s*([kmbtq]?)\b/i);
  if (!stepMatch) return null;
  const metrics = { train: null, evaluation: null };
  const matcher = /([a-z_/-]*loss)\s*[:=]\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)/gi;
  for (const match of line.matchAll(matcher)) {
    const value = Number(match[2]);
    if (!Number.isFinite(value)) continue;
    const series = /eval|valid/i.test(match[1]) ? 'evaluation' : 'train';
    metrics[series] = value;
  }
  const stepScale = { '': 1, k: 1e3, m: 1e6, b: 1e9, t: 1e12, q: 1e15 };
  const step = Number(stepMatch[1].replaceAll(',', '')) * stepScale[stepMatch[2].toLowerCase()];
  return Number.isFinite(step) && (metrics.train !== null || metrics.evaluation !== null) ? { step, ...metrics } : null;
}

function drawLossChart() {
  const root = $('loss-chart');
  root.replaceChildren();
  if (!lossPoints.length) {
    root.append(create('p', 'empty-inline', 'Loss values appear when the training log reports them.'));
    $('loss-state').textContent = 'Waiting for loss metrics';
    return;
  }
  const width = 640, height = 210;
  const plot = { left: 62, right: 624, top: 17, bottom: 166 };
  const series = [
    { key: 'train', label: 'Training loss', color: '#218d86' },
    { key: 'evaluation', label: 'Evaluation loss', color: '#e0944d' },
  ].map(item => ({ ...item, points: lossPoints.filter(point => point[item.key] !== null) })).filter(item => item.points.length);
  const vals = series.flatMap(item => item.points.map(point => point[item.key]));
  const rawMin = Math.min(...vals), rawMax = Math.max(...vals);
  const rawSpan = rawMax - rawMin;
  const padding = rawSpan ? rawSpan * .12 : Math.max(Math.abs(rawMax) * .08, .001);
  const min = rawMin - padding, max = rawMax + padding;
  const firstStep = lossPoints[0].step, lastStep = lossPoints[lossPoints.length - 1].step;
  const x = (step, i) => plot.left + (lastStep === firstStep ? i / Math.max(lossPoints.length - 1, 1) : (step - firstStep) / (lastStep - firstStep)) * (plot.right - plot.left);
  const y = value => plot.bottom - ((value - min) / (max - min)) * (plot.bottom - plot.top);
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`); svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', `Training loss metrics from backend logs${lossDatasetLabel ? ` for ${lossDatasetLabel}` : ''}`);
  const addText = (text, xValue, yValue, className, anchor = 'middle') => {
    const node = document.createElementNS(svg.namespaceURI, 'text');
    node.setAttribute('x', xValue); node.setAttribute('y', yValue);
    node.setAttribute('class', className); node.setAttribute('text-anchor', anchor);
    node.textContent = text; svg.append(node);
  };
  const formatValue = value => Number(value).toPrecision(3).replace(/\.0+(?=e|$)/, '');
  for (let index = 0; index <= 4; index += 1) {
    const fraction = index / 4;
    const line = document.createElementNS(svg.namespaceURI, 'line');
    const tickValue = max - fraction * (max - min);
    const yy = plot.top + fraction * (plot.bottom - plot.top);
    line.setAttribute('x1', plot.left); line.setAttribute('x2', plot.right);
    line.setAttribute('y1', yy); line.setAttribute('y2', yy);
    line.setAttribute('class', 'loss-gridline');
    svg.append(line);
    addText(formatValue(tickValue), plot.left - 8, yy + 3, 'loss-tick-label', 'end');
  }
  const xAxis = document.createElementNS(svg.namespaceURI, 'line');
  xAxis.setAttribute('x1', plot.left); xAxis.setAttribute('x2', plot.right);
  xAxis.setAttribute('y1', plot.bottom); xAxis.setAttribute('y2', plot.bottom);
  xAxis.setAttribute('class', 'loss-axis-line'); svg.append(xAxis);
  const yAxis = document.createElementNS(svg.namespaceURI, 'line');
  yAxis.setAttribute('x1', plot.left); yAxis.setAttribute('x2', plot.left);
  yAxis.setAttribute('y1', plot.top); yAxis.setAttribute('y2', plot.bottom);
  yAxis.setAttribute('class', 'loss-axis-line'); svg.append(yAxis);
  const xTicks = [...new Set([firstStep, Math.round((firstStep + lastStep) / 2), lastStep])];
  xTicks.forEach(step => {
    const xValue = x(step, 0);
    addText(Number(step).toLocaleString(), xValue, plot.bottom + 15, 'loss-tick-label');
  });
  addText('Loss', 13, (plot.top + plot.bottom) / 2, 'loss-axis-label');
  addText('Training step', (plot.left + plot.right) / 2, height - 4, 'loss-axis-label');
  series.forEach(item => {
    const path = document.createElementNS(svg.namespaceURI, 'path');
    path.setAttribute('d', item.points.map((point, index) => `${index ? 'L' : 'M'} ${x(point.step, index).toFixed(1)} ${y(point[item.key]).toFixed(1)}`).join(' '));
    path.setAttribute('fill', 'none'); path.setAttribute('stroke', item.color); path.setAttribute('stroke-width', '2.5'); path.setAttribute('vector-effect', 'non-scaling-stroke');
    svg.append(path);
  });
  root.append(svg);
  const legend = create('div', 'loss-legend');
  series.forEach(item => { const label = create('span', '', item.label); label.dataset.series = item.key; legend.append(label); });
  root.append(legend);
  const runLabel = lossDatasetLabel ? ` · ${lossDatasetLabel}` : '';
  $('loss-state').textContent = `${lossPoints.length} reported step${lossPoints.length === 1 ? '' : 's'}${runLabel}`;
}

function resetLossMetrics(datasetId = '') {
  lossPoints.splice(0);
  lossDatasetLabel = state.datasets.find(item => item.dataset_id === datasetId)?.name || datasetId || '';
  drawLossChart();
}

function appendLogLine(line) {
  const root = $('training-logs');
  if (root.querySelector('.log-placeholder')) root.replaceChildren();
  // Stick to the bottom only when the user is already there; if they have scrolled up
  // to read earlier output, leave their position alone so the terminal stays scrollable
  // while new lines stream in. (A small threshold absorbs sub-pixel rounding.)
  const atBottom = root.scrollHeight - root.scrollTop - root.clientHeight < 40;
  const row = create('div', 'log-line', line);
  root.append(row);
  while (root.childElementCount > 350) root.firstElementChild.remove();
  if (atBottom) root.scrollTop = root.scrollHeight;
  const metric = metricFromLog(line);
  if (metric) {
    const existing = lossPoints.findIndex(point => point.step === metric.step);
    if (existing >= 0) {
      if (metric.train !== null) lossPoints[existing].train = metric.train;
      if (metric.evaluation !== null) lossPoints[existing].evaluation = metric.evaluation;
    }
    else lossPoints.push(metric);
    lossPoints.sort((a, b) => a.step - b.step);
    if (lossPoints.length > 180) lossPoints.splice(0, lossPoints.length - 180);
    drawLossChart();
  }
}

function setStreamState(value) {
  const el = $('training-stream-state');
  el.textContent = value === 'connected' ? 'Connected' : value === 'reconnecting' ? 'Reconnecting' : 'Disconnected';
  el.classList.toggle('connected', value === 'connected');
}

function syncTrainingLog(status) {
  if (status?.active) subscribeLogs('training', '/api/train/logs', appendLogLine, setStreamState);
  else { closeLogStream('training'); setStreamState('disconnected'); }
}

let startBusy = false;
let stopBusy = false;
let resumeBusy = false;
let resetBusy = false;
async function startTraining() {
  if (startBusy) return;
  const dataset = state.datasets.find(item => item.dataset_id === $('train-dataset').value);
  if (!state.selectedPolicy || !dataset) { notice('Choose a policy and dataset before starting training.', 'error'); return; }
  startBusy = true; updateStartAvailability();
  $('train-start-btn').textContent = 'Starting…';
  try {
    const result = await api.trainStart({
      policy_type: state.selectedPolicy,
      dataset_path: dataset.path || dataset.dataset_id,
      config: hyperparameterValues(),
      job_name: $('job-name').value.trim() || undefined,
    });
    resetLossMetrics(dataset.dataset_id); $('training-logs').replaceChildren();
    notice(result.message || 'Training is preparing the dataset.', 'success');
    state.training = { ...(state.training || {}), active: true };
    syncTrainingLog(state.training);
  } catch (error) { notice(error.message, 'error'); }
  finally { startBusy = false; $('train-start-btn').textContent = 'Start training ↗'; updateStartAvailability(); }
}

async function stopTraining() {
  if (stopBusy || $('train-stop-btn').disabled) return;
  stopBusy = true; $('train-stop-btn').disabled = true;
  try { const result = await api.trainStop(); notice(result.message || 'Training stopped.', 'success'); }
  catch (error) { notice(error.message, 'error'); }
  finally { stopBusy = false; updateTrainingStatus(state.training); }
}

async function resetTrainingEnvironment() {
  if (resetBusy) return;
  resetBusy = true;
  const button = $('reset-training-btn');
  button.disabled = true;
  try {
    const status = await api.trainStatus();
    if (status.active && !window.confirm('Stop the active training run and reset this page? Saved checkpoints and output files will be kept.')) return;
    if (status.active) await api.trainStop();
    await api.trainReset();

    closeLogStream('training');
    setStreamState('disconnected');
    $('training-logs').replaceChildren(create('span', 'log-placeholder', 'Logs will appear when the backend emits training output.'));
    resetLossMetrics('');
    $('job-name').value = '';
    $('train-dataset').value = '';
    state.selectedDatasetId = '';
    pendingResumeRun = null;
    pendingResumePolicy = '';
    try { window.localStorage.removeItem('daksha.selectedPolicy'); } catch (_) { /* selection remains in memory */ }
    const defaultPolicy = Object.keys(state.policies || {})[0];
    if (defaultPolicy) setSelectedPolicy(defaultPolicy);
    $('train-dataset').dispatchEvent(new Event('change', { bubbles: true }));
    updateTrainingStatus(await api.trainStatus());
    notice('Training environment reset. Saved checkpoints and output files were kept.', 'success');
  } catch (error) {
    notice(error.message, 'error');
  } finally {
    resetBusy = false;
    button.disabled = false;
    updateStartAvailability();
  }
}

function normalizeDatasetRoot(path) {
  return String(path || '').replaceAll('\\', '/').replace(/\/+$/, '').toLowerCase();
}

function datasetForRun(run, datasets = state.datasets) {
  const checkpointDataset = normalizeDatasetRoot(run?.dataset_root);
  const task = normalizeDatasetRoot(run?.task);
  return datasets.find(item =>
    (checkpointDataset && normalizeDatasetRoot(item.path) === checkpointDataset) ||
    (task && (normalizeDatasetRoot(item.dataset_id) === task || normalizeDatasetRoot(item.name) === task)) ||
    (checkpointDataset && checkpointDataset.split('/').at(-1) === normalizeDatasetRoot(item.dataset_id))
  );
}

function resetTrainingForCheckpoint(run) {
  // The checkpoint's saved config controls the actual backend resume. Mirror its
  // policy and dataset in the UI so the video sample and run labels match it.
  const requestedPolicy = String(run?.model || '').toLowerCase();
  const policyAlias = requestedPolicy === 'diffusion_policy' ? 'diffusion' : requestedPolicy;
  const policyKey = Object.keys(state.policies || {}).find(key => key.toLowerCase() === policyAlias);
  if (policyKey) { pendingResumePolicy = ''; setSelectedPolicy(policyKey); }
  else pendingResumePolicy = policyAlias;

  const dataset = datasetForRun(run);
  pendingResumeRun = dataset ? null : run;

  state.training = { ...(state.training || {}), active: false };
  closeLogStream('training');
  lossPoints.splice(0);
  $('training-logs').replaceChildren(create('span', 'log-placeholder', 'Connecting to resumed training logs…'));
  resetLossMetrics(dataset?.dataset_id || run?.task || '');
  $('job-name').value = run?.name || '';
  if (dataset) {
    $('train-dataset').value = dataset.dataset_id;
    state.selectedDatasetId = dataset.dataset_id;
    $('train-dataset').dispatchEvent(new Event('change', { bubbles: true }));
  } else {
    $('train-dataset').value = '';
    state.selectedDatasetId = '';
    $('train-dataset').dispatchEvent(new Event('change', { bubbles: true }));
    $('sample-state').textContent = run?.dataset_root
      ? `Dataset ${run.dataset_root} is not in the current catalog.`
      : 'Dataset metadata is unavailable for this checkpoint.';
  }
}

async function resumeTraining(checkpointPath = '', run = null, trigger = null) {
  if (resumeBusy) return;
  resumeBusy = true; $('train-resume-btn').disabled = true;
  if (checkpointPath) {
    window.dispatchEvent(new CustomEvent('app:navigate', { detail: 'training' }));
    resetTrainingForCheckpoint(run);
  }
  if (trigger) { trigger.disabled = true; trigger.textContent = 'Resuming…'; }
  try {
    const result = await api.trainResume(checkpointPath);
    lossPoints.splice(0);
    resetLossMetrics(state.selectedDatasetId || run?.task || '');
    // Reflect the resumed run's policy and original training parameters in the UI so the
    // Training-configuration panel matches what the backend restored from train_config.json.
    if (result.policy_type) {
      const key = Object.keys(state.policies || {}).find(k => k.toLowerCase() === String(result.policy_type).toLowerCase());
      if (key && key !== state.selectedPolicy) setSelectedPolicy(key);
    }
    applyHyperparameterValues(result.hyperparameters);
    $('training-logs').replaceChildren(create('span', 'log-placeholder', 'Waiting for resumed training output…'));
    state.training = {
      ...(state.training || {}), active: true, stage: result.stage || 'preparing',
      policy: run?.model || state.selectedPolicy,
      current_step: Number(result.resumed_from_step || 0),
      total_steps: Number(result.total_steps || 0),
      progress: result.total_steps ? Number(result.resumed_from_step || 0) / Number(result.total_steps) * 100 : 0,
    };
    updateTrainingStatus(state.training);
    notice(result.message || 'Training resumed.', 'success');
  } catch (error) {
    notice(error.message, 'error');
    try { updateTrainingStatus(await api.trainStatus()); } catch (_) { /* retain the reset UI if status polling is unavailable */ }
  } finally {
    resumeBusy = false;
    if (trigger) {
      trigger.disabled = Boolean(state.training?.active);
      trigger.textContent = state.training?.active ? 'Running' : 'Resume run';
    }
    updateTrainingStatus(state.training);
  }
}

function updateTrainingStatus(status) {
  if (!status) return;
  state.training = status;
  const active = Boolean(status.active);
  const stage = status.stage || (active ? 'starting' : 'idle');
  const label = stage === 'converting' || stage === 'preparing' ? 'Preparing dataset' : stage === 'training' || active ? 'Training' : 'Idle';
  $('train-stage').textContent = active ? `${label}${status.policy ? ` · ${status.policy}` : ''}` : 'Waiting for a run';
  $('train-step').textContent = status.total_steps ? `${Number(status.current_step || 0).toLocaleString()} / ${Number(status.total_steps).toLocaleString()} steps` : active ? 'Preparing' : '—';
  const progress = Math.max(0, Math.min(100, Number(status.progress || 0)));
  $('train-progress').textContent = status.total_steps ? `${progress.toFixed(1)}%` : active ? 'Preparing' : '—';
  $('progress-fill').style.width = `${progress}%`;
  const bar = $('progress-fill').parentElement;
  bar.setAttribute('aria-valuenow', String(Math.round(progress)));
  $('train-stop-btn').disabled = !active || stopBusy;
  $('train-resume-btn').disabled = active || !status.retrain?.available || resumeBusy;
  updateStartAvailability();
  syncTrainingLog(status);
}

function setSelect(select, entries, placeholder, valueOf, labelOf) {
  select.replaceChildren(new Option(placeholder, ''));
  entries.forEach(item => select.add(new Option(labelOf(item), String(valueOf(item)))));
}

function currentSampleEpisode() {
  const id = Number($('sample-episode').value);
  return sampleEpisodes.find(item => Number(item.episode_index) === id) || null;
}

function updateSampleCameraOptions(preferredKey = '') {
  const episode = currentSampleEpisode();
  const select = $('sample-camera');
  const videos = (episode?.videos || []).filter(video => video.available && video.url);
  setSelect(select, videos, videos.length ? 'Select camera' : 'No recorded video', item => item.camera_key, item => item.label || item.camera_key);
  if (videos.length) select.value = videos.some(item => item.camera_key === preferredKey) ? preferredKey : videos[0].camera_key;
  select.disabled = !videos.length;
  $('sample-state').textContent = videos.length ? 'Camera recording available' : 'No camera media for this episode';
  updateTrainingSampleVideo();
}

function updateTrainingSampleVideo() {
  const episode = currentSampleEpisode();
  const video = $('training-sample-video');
  const selected = episode?.videos?.find(item => item.camera_key === $('sample-camera').value && item.available && item.url);
  video.pause();
  if (!selected) {
    sampleSelected = null;
    video.removeAttribute('src'); video.load();
    $('sample-empty').hidden = false;
    ['sample-play', 'sample-pause', 'sample-stop'].forEach(id => { $(id).disabled = true; });
    return;
  }
  sampleSelected = selected;
  if (video.src !== new URL(selected.url, window.location.href).href) {
    video.src = selected.url;
    video.load();
  } else if (video.readyState >= 1) {
    video.currentTime = Number(selected.start_time || 0);
  }
  $('sample-empty').hidden = true;
  $('sample-play').disabled = false; $('sample-pause').disabled = false; $('sample-stop').disabled = false;
}

async function refreshTrainingSample(datasetId) {
  const token = ++sampleToken;
  sampleMetadata = null; sampleEpisodes = [];
  const episodeSelect = $('sample-episode');
  const cameraSelect = $('sample-camera');
  episodeSelect.replaceChildren(new Option('Loading episodes…', '')); episodeSelect.disabled = true;
  cameraSelect.replaceChildren(new Option('No camera', '')); cameraSelect.disabled = true;
  updateTrainingSampleVideo();
  if (!datasetId) { $('sample-state').textContent = 'Select a dataset'; return; }
  $('sample-state').textContent = 'Loading dataset metadata…';
  try {
    const [metadata, result] = await Promise.all([api.viewer(datasetId), api.episodes(datasetId, { limit: 120 })]);
    if (token !== sampleToken) return;
    sampleMetadata = metadata; sampleEpisodes = result.episodes || [];
    setSelect(episodeSelect, sampleEpisodes, sampleEpisodes.length ? 'Select episode' : 'No episodes', item => item.episode_index, item => `Episode ${item.episode_index} · ${item.frame_count} frames`);
    episodeSelect.disabled = sampleEpisodes.length === 0;
    $('sample-state').textContent = sampleEpisodes.length ? `${metadata.fps || '—'} FPS · sample from catalog` : 'No episodes available';
    if (sampleEpisodes.length) {
      episodeSelect.value = String(sampleEpisodes[0].episode_index);
      updateSampleCameraOptions(metadata.cameras?.find(item => item.available)?.key || '');
    }
  } catch (error) {
    if (token !== sampleToken) return;
    $('sample-state').textContent = 'Dataset sample unavailable';
    $('sample-empty').hidden = false;
    $('sample-empty').textContent = error.message;
  }
}

export function initTraining() {
  $('train-dataset').addEventListener('change', event => {
    state.selectedDatasetId = event.target.value;
    if (!state.training?.active) resetLossMetrics(event.target.value);
    renderSelectedDatasetInfo(); updateStartAvailability(); refreshTrainingSample(event.target.value);
  });
  $('refresh-datasets-training').addEventListener('click', () => window.dispatchEvent(new Event('catalog:refresh-datasets')));
  $('reset-training-btn').addEventListener('click', resetTrainingEnvironment);
  $('train-start-btn').addEventListener('click', startTraining);
  $('train-stop-btn').addEventListener('click', stopTraining);
  $('train-resume-btn').addEventListener('click', () => resumeTraining());
  $('sample-episode').addEventListener('change', () => updateSampleCameraOptions());
  $('sample-camera').addEventListener('change', updateTrainingSampleVideo);
  $('sample-play').addEventListener('click', () => {
    const video = $('training-sample-video');
    if (sampleSelected && Number.isFinite(Number(sampleSelected.end_time)) && video.currentTime >= Number(sampleSelected.end_time) - .04) video.currentTime = Number(sampleSelected.start_time || 0);
    video.play().catch(error => notice(`Video playback failed: ${error.message}`, 'error'));
  });
  $('sample-pause').addEventListener('click', () => $('training-sample-video').pause());
  $('sample-stop').addEventListener('click', () => { const video = $('training-sample-video'); video.pause(); if (video.readyState) video.currentTime = Number(sampleSelected?.start_time || 0); });
  $('training-sample-video').addEventListener('loadedmetadata', () => {
    $('sample-empty').hidden = true;
    if (sampleSelected) $('training-sample-video').currentTime = Number(sampleSelected.start_time || 0);
  });
  $('training-sample-video').addEventListener('timeupdate', () => {
    const video = $('training-sample-video');
    if (sampleSelected && Number.isFinite(Number(sampleSelected.end_time)) && video.currentTime >= Number(sampleSelected.end_time)) {
      video.pause(); video.currentTime = Math.max(Number(sampleSelected.start_time || 0), Number(sampleSelected.end_time) - .03);
    }
  });
  $('training-sample-video').addEventListener('error', () => {
    $('sample-empty').hidden = false;
    $('sample-empty').textContent = 'This episode video could not be loaded.';
  });
  window.addEventListener('catalog:datasets', event => updateTrainingDatasetOptions(event.detail || []));
  window.addEventListener('training:status', event => updateTrainingStatus(event.detail));
  try { state.selectedPolicy = window.localStorage.getItem('daksha.selectedPolicy') || state.selectedPolicy; } catch (_) { /* selection remains in memory */ }
}

export function loadPolicies(policies) {
  state.policies = policies || {};
  if (pendingResumePolicy) {
    const key = Object.keys(state.policies).find(item => item.toLowerCase() === pendingResumePolicy);
    if (key) {
      state.selectedPolicy = key;
      try { window.localStorage.setItem('daksha.selectedPolicy', key); } catch (_) { /* selection remains in memory */ }
      pendingResumePolicy = '';
    }
  }
  renderPolicies();
}

export function setSelectedPolicy(policyKey) {
  if (!state.policies[policyKey]) return;
  state.selectedPolicy = policyKey;
  try { window.localStorage.setItem('daksha.selectedPolicy', policyKey); } catch (_) { /* selection remains in memory */ }
  renderPolicies(); updateStartAvailability();
}

export { resumeTraining };
