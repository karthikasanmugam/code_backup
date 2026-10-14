import { api } from './api.js';
import { state } from './state.js';
import { resumeTraining } from './training.js';

const $ = id => document.getElementById(id);
const el = (tag, className = '', text = '') => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== '') node.textContent = text;
  return node;
};
const notice = (message, kind = 'info') => window.dispatchEvent(new CustomEvent('app:notice', { detail: { message, kind } }));

function formatBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes < 0) return 'Size unavailable';
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const step = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** step).toFixed(step > 1 ? 2 : 0)} ${units[step]}`;
}

function addCheckpointOption(select, run, item) {
  const label = `${run.model || 'Policy'} · ${run.task || run.name} · ${item.label || item.path}`;
  select.add(new Option(label, item.path));
}

function renderRun(run) {
  const card = el('article', `checkpoint-card checkpoint-${run.model || 'policy'}`);
  const title = el('div', 'item-title');
  title.append(el('strong', '', `${String(run.model || 'Policy').toUpperCase()} · ${run.task || run.name || 'Run'}`), el('small', '', run.name || 'Unnamed run'));
  const meta = el('div', 'item-meta');
  const steps = Array.isArray(run.steps) && run.steps.length ? `${run.steps.length} saved step${run.steps.length === 1 ? '' : 's'} · latest ${run.latest_step ?? '—'}` : 'No complete saved steps';
  const created = run.created ? new Date(run.created).toLocaleString() : 'Date unavailable';
  meta.append(el('div', '', `${formatBytes(run.size_bytes)} · ${created}`), el('div', '', steps));
  const actions = el('div', 'item-actions');
  const stepSelect = document.createElement('select');
  stepSelect.className = 'checkpoint-step-select';
  stepSelect.setAttribute('aria-label', `Saved checkpoint step for ${run.name || run.task || 'run'}`);
  const options = Array.isArray(run.checkpoint_options) ? run.checkpoint_options : [];
  stepSelect.add(new Option(options.length ? 'Select exact step' : 'No complete checkpoint', ''));
  options.forEach(item => addCheckpointOption(stepSelect, run, item));
  if (options.length) {
    const load = el('button', 'button secondary compact', 'Use for inference'); load.type = 'button'; load.disabled = true;
    stepSelect.addEventListener('change', () => { load.disabled = !stepSelect.value; });
    load.addEventListener('click', () => {
      window.dispatchEvent(new CustomEvent('checkpoint:select', { detail: { path: stepSelect.value, run } }));
      notice('Checkpoint selected for inference.', 'success');
    });
    actions.append(stepSelect, load);
  } else actions.append(stepSelect);
  const resume = el('button', 'button secondary compact', 'Resume run'); resume.type = 'button';
  resume.disabled = !run.path;
  resume.addEventListener('click', () => resumeTraining(run.path, run, resume));
  actions.append(resume);
  if (run.can_delete !== false) {
    const remove = el('button', 'button danger compact', 'Delete'); remove.type = 'button';
    remove.addEventListener('click', () => deleteRun(run, remove));
    actions.append(remove);
  }
  card.append(title, meta, actions);
  return card;
}

export async function loadCheckpoints() {
  const list = $('checkpoints-list');
  list.replaceChildren(el('p', 'muted', 'Loading checkpoints…'));
  try {
    const records = await api.checkpoints();
    state.checkpoints = Array.isArray(records) ? records : [];
    $('checkpoint-count').textContent = `${state.checkpoints.length} run${state.checkpoints.length === 1 ? '' : 's'}`;
    list.replaceChildren();
    if (!state.checkpoints.length) list.append(el('div', 'empty-state', 'No checkpoints were found in the configured output roots.'));
    else state.checkpoints.forEach(run => list.append(renderRun(run)));
    window.dispatchEvent(new CustomEvent('catalog:checkpoints', { detail: state.checkpoints }));
    return state.checkpoints;
  } catch (error) {
    list.replaceChildren(el('div', 'empty-state', `Could not load checkpoints: ${error.message}`));
    $('checkpoint-count').textContent = 'Catalog unavailable'; notice(error.message, 'error'); return [];
  }
}

async function deleteRun(run, button) {
  if (!window.confirm(`Permanently delete checkpoint run “${run.name}”?`)) return;
  button.disabled = true;
  try {
    const result = await api.checkpointDelete(run.path);
    notice(result.message || 'Checkpoint deleted.', 'success');
    await loadCheckpoints();
  } catch (error) { notice(error.message, 'error'); button.disabled = false; }
}

export function initCheckpoints() {
  $('refresh-checkpoints').addEventListener('click', loadCheckpoints);
  $('refresh-checkpoints-inference').addEventListener('click', loadCheckpoints);
  window.addEventListener('catalog:refresh-checkpoints', loadCheckpoints);
}
