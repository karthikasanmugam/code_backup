import { api, closeLogStream, subscribeLogs } from './api.js';
import { state } from './state.js';

const $ = id => document.getElementById(id);
const el = (tag, className = '', text = '') => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== '') node.textContent = text;
  return node;
};
const notice = (message, kind = 'info') => window.dispatchEvent(new CustomEvent('app:notice', { detail: { message, kind } }));
let metadataBusy = false;
let serverBusy = false;
let serverAction = '';
let bridgeBusy = false;
let resetBusy = false;
let serverCheckpointPath = '';
let metadataTimer = null;
let metadataRequest = 0;

function checkpointOptions(records, loadSelectedMetadata = true) {
  const filter = $('inf-policy-filter').value;
  const select = $('inf-checkpoint');
  const prior = state.selectedCheckpointPath || select.value;
  const all = [];
  records.forEach(run => {
    if (filter && run.model !== filter) return;
    (run.checkpoint_options || []).forEach(option => all.push({ run, option }));
  });
  select.replaceChildren(new Option(all.length ? 'Select an exact saved step' : 'No saved steps available', ''));
  all.forEach(({ run, option }) => {
    const policy = String(run.model || 'policy').toUpperCase();
    const label = `${policy} · ${run.task || run.name} · ${option.label || option.path}`;
    select.add(new Option(label, option.path));
  });
  if (all.some(({ option }) => option.path === prior)) select.value = prior;
  else if (state.selectedCheckpointPath && filter) state.selectedCheckpointPath = '';
  $('server-start-btn').disabled = !select.value || serverBusy;
  if (select.value && loadSelectedMetadata) loadInferenceMetadata(true);
}

function setChip(id, active, label) {
  const chip = $(id);
  chip.classList.toggle('is-active', Boolean(active));
  chip.querySelector('span').textContent = label;
}

function setBridgeSummary(status) {
  const root = $('bridge-checkpoint-info');
  root.replaceChildren();
  if (!state.selectedCheckpointPath) {
    root.textContent = 'Select a checkpoint and load its metadata first.';
    return;
  }
  const meta = state.inferenceMetadata || {};
  const checkpoint = $('inf-checkpoint').selectedOptions[0]?.textContent || state.selectedCheckpointPath;
  const badge = el('span', `status-badge ${status === 'loaded' ? 'good' : ''}`, status === 'loaded' ? 'LOADED' : 'SELECTED');
  const title = el('strong', '', checkpoint);
  const detail = el('p', '', `Policy ${meta.policy_type || '—'} · ${meta.dataset_fps ?? '—'} FPS · horizon ${meta.action_horizon ?? '—'} / ${meta.chunk_size ?? '—'} · ${meta.control_hz ?? '—'} Hz`);
  root.append(badge, title, detail);
}

function renderLimits(meta) {
  const root = $('inf-limits-body'); root.replaceChildren();
  const names = Array.isArray(meta.action_names) ? meta.action_names : [];
  const mins = String(meta.joint_min_csv || '').split(',').map(Number);
  const maxs = String(meta.joint_max_csv || '').split(',').map(Number);
  const steps = String(meta.max_step_delta_csv || '').split(',').map(Number);
  if (!names.length || mins.length < names.length || maxs.length < names.length || steps.length < names.length) {
    const row = root.insertRow(); const cell = row.insertCell(); cell.colSpan = 5; cell.className = 'muted'; cell.textContent = 'Checkpoint metadata does not include complete safety limits.'; return;
  }
  names.forEach((name, index) => {
    const row = root.insertRow();
    [index, name, mins[index], maxs[index], steps[index]].forEach((value, column) => {
      const cell = row.insertCell();
      cell.textContent = column < 2 ? String(value) : Number.isFinite(Number(value)) ? Number(value).toFixed(4) : '—';
    });
  });
}

async function loadInferenceMetadata(fresh = false) {
  const requestId = ++metadataRequest;
  const checkpoint = $('inf-checkpoint').value;
  if (!checkpoint) {
    state.selectedCheckpointPath = ''; state.inferenceMetadata = null; serverCheckpointPath = '';
    $('inf-detected').textContent = 'Choose a checkpoint to inspect its runtime settings.';
    $('inf-limits-body').replaceChildren();
    const row = $('inf-limits-body').insertRow(); const cell = row.insertCell(); cell.colSpan = 5; cell.className = 'muted'; cell.textContent = 'Select a checkpoint to calculate its limits.';
    setBridgeSummary('none'); updateButtonStates(); return;
  }
  metadataBusy = true; state.selectedCheckpointPath = checkpoint; state.inferenceMetadata = null;
  $('server-start-btn').disabled = true; $('inf-detected').textContent = 'Loading checkpoint metadata…';
  if (fresh) { $('inf-action-horizon').value = ''; $('inf-control-hz').value = ''; }
  try {
    const meta = await api.inferenceMetadata(checkpoint, $('inf-action-horizon').value, $('inf-control-hz').value);
    if (requestId !== metadataRequest || $('inf-checkpoint').value !== checkpoint) return;
    state.inferenceMetadata = meta;
    $('inf-detected').textContent = `Detected ${meta.policy_type || 'unknown'} · dataset ${meta.dataset_fps ?? '—'} FPS · action chunk ${meta.chunk_size ?? '—'}`;
    if (fresh || !$('inf-action-horizon').value) $('inf-action-horizon').value = meta.action_horizon ?? '';
    if (fresh || !$('inf-control-hz').value) $('inf-control-hz').value = meta.control_hz ?? '';
    $('inf-action-horizon').max = meta.chunk_size || '';
    renderLimits(meta); setBridgeSummary(serverCheckpointPath === checkpoint ? 'loaded' : 'selected');
  } catch (error) {
    if (requestId !== metadataRequest || $('inf-checkpoint').value !== checkpoint) return;
    state.inferenceMetadata = null;
    $('inf-detected').textContent = error.message;
    const root = $('inf-limits-body'); root.replaceChildren(); const row = root.insertRow(); const cell = row.insertCell(); cell.colSpan = 5; cell.className = 'muted'; cell.textContent = error.message;
    setBridgeSummary('none');
  } finally {
    if (requestId === metadataRequest) { metadataBusy = false; updateButtonStates(); }
  }
}

function scheduleMetadataRefresh() {
  clearTimeout(metadataTimer);
  metadataTimer = setTimeout(() => loadInferenceMetadata(false), 250);
}

function setLogState(name, value) {
  const node = $(`${name}-stream-state`);
  node.textContent = value === 'connected' ? 'Connected' : value === 'reconnecting' ? 'Reconnecting' : 'Disconnected';
  node.classList.toggle('connected', value === 'connected');
}

function appendLog(name, line) {
  const root = $(`${name}-logs`);
  const placeholder = root.querySelector('.log-placeholder'); if (placeholder) placeholder.remove();
  const row = el('div', 'log-line', line); root.append(row);
  while (root.childElementCount > 250) root.firstElementChild.remove();
  root.scrollTop = root.scrollHeight;
}

function syncLogs(status) {
  const diffusionReady = state.inferenceMetadata?.policy_type === 'diffusion' && serverCheckpointPath === state.selectedCheckpointPath;
  if (status?.inference?.server_running || diffusionReady) subscribeLogs('server', '/api/inference/logs', line => appendLog('server', line), value => setLogState('server', value));
  else { closeLogStream('server'); setLogState('server', 'disconnected'); }
  if (status?.inference?.bridge_running) subscribeLogs('bridge', '/api/inference/bridge_logs', line => appendLog('bridge', line), value => setLogState('bridge', value));
  else { closeLogStream('bridge'); setLogState('bridge', 'disconnected'); }
}

function updateRuntimeLabels() {
  const directLoad = state.inferenceMetadata?.policy_type === 'diffusion';
  $('server-start-btn').textContent = serverBusy
    ? (serverAction === 'start' ? (directLoad ? 'Preparing checkpoint…' : 'Starting policy server…') : (directLoad ? 'Clearing checkpoint…' : 'Stopping server…'))
    : (directLoad ? 'Prepare checkpoint for bridge' : 'Start policy server');
  $('server-stop-btn').textContent = serverBusy && serverAction === 'stop'
    ? (directLoad ? 'Clearing…' : 'Stopping…')
    : (directLoad ? 'Clear loaded checkpoint' : 'Stop server');
}

function updateButtonStates() {
  const hasCheckpoint = Boolean($('inf-checkpoint').value);
  const loadedReady = Boolean(state.inferenceMetadata && serverCheckpointPath === state.selectedCheckpointPath);
  const runtimeReady = state.serverRunning || loadedReady;
  const selectionLocked = runtimeReady || state.bridgeRunning || serverBusy || bridgeBusy;
  $('inf-policy-filter').disabled = selectionLocked;
  $('inf-checkpoint').disabled = selectionLocked;
  $('inf-action-horizon').disabled = selectionLocked;
  $('inf-control-hz').disabled = selectionLocked;
  $('server-port').disabled = runtimeReady || serverBusy;
  $('bridge-device').disabled = runtimeReady || state.bridgeRunning || serverBusy || bridgeBusy;
  updateRuntimeLabels();
  $('server-start-btn').disabled = !hasCheckpoint || runtimeReady || serverBusy || metadataBusy;
  $('server-stop-btn').disabled = !runtimeReady || serverBusy;
  const metaReady = Boolean(state.inferenceMetadata && serverCheckpointPath === state.selectedCheckpointPath);
  const policy = state.inferenceMetadata?.policy_type;
  const bridgePrerequisite = metaReady && (state.serverRunning || policy === 'diffusion');
  $('bridge-start-btn').disabled = !bridgePrerequisite || state.bridgeRunning || bridgeBusy;
  $('bridge-stop-btn').disabled = !state.bridgeRunning || bridgeBusy;
  $('reset-inference-btn').disabled = resetBusy || serverBusy || bridgeBusy;
}

export function updateInferenceStatus(status) {
  if (!status) return;
  state.serverRunning = Boolean(status.inference?.server_running);
  state.bridgeRunning = Boolean(status.inference?.bridge_running);
  const loadedReady = Boolean(state.inferenceMetadata && serverCheckpointPath === state.selectedCheckpointPath);
  const directReady = loadedReady && state.inferenceMetadata?.policy_type === 'diffusion';
  setChip('server-state-chip', state.serverRunning || loadedReady, state.serverRunning ? `Online${status.inference?.server_port ? ` · ${status.inference.server_port}` : ''}` : directReady ? 'Direct load ready' : loadedReady ? 'Loading' : 'Offline');
  setChip('bridge-state-chip', state.bridgeRunning, state.bridgeRunning ? 'Running' : 'Offline');
  syncLogs(status); updateButtonStates();
}

async function startServer() {
  if (serverBusy) return;
  const checkpoint = $('inf-checkpoint').value;
  if (!checkpoint) return;
  serverBusy = true; serverAction = 'start'; updateButtonStates();
  try {
    const result = await api.serverStart({
      checkpoint_path: checkpoint,
      port: Number($('server-port').value) || undefined,
      device: $('bridge-device').value,
      action_horizon: $('inf-action-horizon').value ? Number($('inf-action-horizon').value) : undefined,
      control_hz: $('inf-control-hz').value ? Number($('inf-control-hz').value) : undefined,
    });
    serverCheckpointPath = checkpoint;
    notice(result.message || 'Policy runtime is starting.', 'success');
    if (result.port) $('server-port').value = result.port;
    setBridgeSummary('loaded');
    updateButtonStates();
  } catch (error) { notice(error.message, 'error'); }
  finally { serverBusy = false; serverAction = ''; updateButtonStates(); }
}

async function stopServer() {
  const loadedReady = Boolean(state.inferenceMetadata && serverCheckpointPath === state.selectedCheckpointPath);
  if (serverBusy || (!state.serverRunning && !loadedReady)) return;
  serverBusy = true; serverAction = 'stop'; updateButtonStates();
  try {
    const result = await api.serverStop();
    state.serverRunning = false; state.bridgeRunning = false; serverCheckpointPath = '';
    notice(result.message || 'Policy server stopped.', 'success');
    setBridgeSummary('selected');
  } catch (error) { notice(error.message, 'error'); }
  finally { serverBusy = false; serverAction = ''; updateButtonStates(); }
}

async function startBridge() {
  if (bridgeBusy) return;
  bridgeBusy = true; updateButtonStates(); $('bridge-start-btn').textContent = 'Starting…';
  try {
    const result = await api.bridgeStart({ device: $('bridge-device').value });
    notice(result.message || 'Robot bridge is starting.', 'success');
  } catch (error) { notice(error.message, 'error'); }
  finally { bridgeBusy = false; $('bridge-start-btn').textContent = 'Start robot bridge'; updateButtonStates(); }
}

async function stopBridge() {
  if (bridgeBusy || !state.bridgeRunning) return;
  bridgeBusy = true; updateButtonStates();
  try { const result = await api.bridgeStop(); notice(result.message || 'Robot bridge stopped.', 'success'); }
  catch (error) { notice(error.message, 'error'); }
  finally { bridgeBusy = false; updateButtonStates(); }
}

async function resetInferenceEnvironment() {
  if (resetBusy || serverBusy || bridgeBusy) return;
  resetBusy = true; updateButtonStates();
  try {
    const status = await api.status();
    const active = Boolean(
      status.inference?.server_running || status.inference?.bridge_running
      || state.serverRunning || state.bridgeRunning || serverCheckpointPath
    );
    if (active && !window.confirm('Stop the active inference runtime and robot bridge, then reset this page? Saved checkpoints and output files will be kept.')) return;
    if (active) await api.serverStop();

    closeLogStream('server'); closeLogStream('bridge');
    setLogState('server', 'disconnected'); setLogState('bridge', 'disconnected');
    $('server-logs').replaceChildren(el('span', 'log-placeholder', 'Logs will appear when the policy runtime emits output.'));
    $('bridge-logs').replaceChildren(el('span', 'log-placeholder', 'Logs will appear when the bridge emits output.'));
    state.serverRunning = false; state.bridgeRunning = false;
    serverCheckpointPath = '';
    $('inf-policy-filter').value = '';
    $('inf-checkpoint').value = '';
    $('inf-action-horizon').value = '';
    $('inf-control-hz').value = '';
    $('inf-detected').textContent = 'Choose a checkpoint to inspect its runtime settings.';
    await loadInferenceMetadata(true);
    updateInferenceStatus({ inference: { server_running: false, bridge_running: false } });
    notice('Inference environment reset. Saved checkpoints and output files were kept.', 'success');
  } catch (error) { notice(error.message, 'error'); }
  finally { resetBusy = false; updateButtonStates(); }
}

export function initInference() {
  $('inf-policy-filter').addEventListener('change', () => checkpointOptions(state.checkpoints));
  $('inf-checkpoint').addEventListener('change', () => {
    serverCheckpointPath = '';
    loadInferenceMetadata(true);
  });
  $('inf-action-horizon').addEventListener('change', scheduleMetadataRefresh);
  $('inf-control-hz').addEventListener('change', scheduleMetadataRefresh);
  $('server-start-btn').addEventListener('click', startServer);
  $('server-stop-btn').addEventListener('click', stopServer);
  $('bridge-start-btn').addEventListener('click', startBridge);
  $('bridge-stop-btn').addEventListener('click', stopBridge);
  $('reset-inference-btn').addEventListener('click', resetInferenceEnvironment);
  window.addEventListener('catalog:checkpoints', event => checkpointOptions(event.detail || []));
  window.addEventListener('system:status', event => updateInferenceStatus(event.detail));
  window.addEventListener('checkpoint:select', event => {
    const { path, run } = event.detail || {};
    if (run?.model) $('inf-policy-filter').value = run.model;
    checkpointOptions(state.checkpoints, false);
    $('inf-checkpoint').value = path || '';
    if (path) { serverCheckpointPath = ''; loadInferenceMetadata(true); window.dispatchEvent(new CustomEvent('app:navigate', { detail: 'inference' })); }
  });
}
