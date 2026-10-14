import { api } from './api.js';
import { state } from './state.js';

const $ = id => document.getElementById(id);
const el = (tag, className = '', text = '') => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== '') node.textContent = text;
  return node;
};
const notice = (message, kind = 'info') => window.dispatchEvent(new CustomEvent('app:notice', { detail: { message, kind } }));
let preparationGeneration = 0;
let cancelPreparationDelay = null;

function renderDatasetCard(dataset) {
  const card = el('article', 'dataset-card');
  const title = el('div', 'item-title');
  title.append(el('strong', '', dataset.name || dataset.dataset_id), el('small', '', dataset.dataset_id));
  const meta = el('div', 'item-meta');
  const size = Number.isFinite(Number(dataset.size_gb)) ? `${dataset.size_gb} GB` : 'Size unavailable';
  const created = dataset.created ? new Date(dataset.created).toLocaleString() : 'Date unavailable';
  meta.append(el('div', '', size), el('div', '', created));
  const actions = el('div', 'item-actions');
  const validate = el('button', 'button secondary compact', 'Validate'); validate.type = 'button';
  const view = el('button', 'button primary compact', 'View episodes'); view.type = 'button';
  const remove = el('button', 'button danger compact', 'Delete'); remove.type = 'button';
  validate.addEventListener('click', () => prepareAndValidate(dataset, validate));
  view.addEventListener('click', () => prepareAndView(dataset, view));
  remove.addEventListener('click', () => deleteDataset(dataset, remove));
  actions.append(validate, view, remove);
  card.append(title, meta, actions);
  return card;
}

export async function loadDatasets() {
  const list = $('datasets-list');
  list.replaceChildren(el('p', 'muted', 'Loading the configured dataset catalog…'));
  try {
    const datasets = await api.datasets();
    state.datasets = Array.isArray(datasets) ? datasets : [];
    $('dataset-count').textContent = `${state.datasets.length} dataset${state.datasets.length === 1 ? '' : 's'}`;
    list.replaceChildren();
    if (!state.datasets.length) list.append(el('div', 'empty-state', 'No datasets were found in the configured dataset root.'));
    else state.datasets.forEach(dataset => list.append(renderDatasetCard(dataset)));
    window.dispatchEvent(new CustomEvent('catalog:datasets', { detail: state.datasets }));
    return state.datasets;
  } catch (error) {
    list.replaceChildren(el('div', 'empty-state', `Could not load datasets: ${error.message}`));
    $('dataset-count').textContent = 'Catalog unavailable';
    notice(error.message, 'error');
    return [];
  }
}

function setLoading(progress, stage, message) {
  const percent = Math.max(0, Math.min(100, Number(progress) || 0));
  $('dataset-loading-fill').style.width = `${percent}%`;
  $('dataset-loading').querySelector('[role="progressbar"]').setAttribute('aria-valuenow', String(Math.round(percent)));
  $('dataset-loading-percent').textContent = `${Math.round(percent)}%`;
  $('dataset-loading-stage').textContent = stage || 'Preparing dataset…';
  $('dataset-loading-message').textContent = message || 'Initial dataset loading will take time. Please keep this page open.';
  $('dataset-loading').hidden = false;
}

function closeLoading() { $('dataset-loading').hidden = true; }

async function prepareDataset(dataset, purpose) {
  const generation = ++preparationGeneration;
  const close = $('dataset-loading-close');
  close.hidden = true;
  $('dataset-loading-title').textContent = 'Preparing dataset';
  setLoading(0, 'Starting dataset checks…');
  try {
    const { job_id: jobId } = await api.prepareDataset(dataset.dataset_id, purpose);
    if (generation !== preparationGeneration) return { cancelled: true };
    if (!jobId) throw new Error('The dataset preparation job could not be started.');
    let result;
    while (true) {
      await new Promise(resolve => {
        const timer = window.setTimeout(resolve, 850);
        cancelPreparationDelay = () => { window.clearTimeout(timer); resolve(); };
      });
      cancelPreparationDelay = null;
      if (generation !== preparationGeneration) return { cancelled: true };
      result = await api.datasetPreparation(jobId);
      if (generation !== preparationGeneration) return { cancelled: true };
      const stage = result.status === 'complete'
        ? (purpose === 'validate' ? 'Opening validation tools…' : 'Opening dataset viewer…')
        : (result.message || 'Preparing dataset…').replace(/convert(?:ing|ed|sion)?/ig, 'preparing');
      setLoading(result.progress, stage);
      if (result.status === 'complete') return result;
      if (result.status === 'error') throw new Error(result.message || 'Dataset preparation failed.');
    }
  } catch (error) {
    if (generation !== preparationGeneration) return { cancelled: true };
    $('dataset-loading-title').textContent = 'Dataset could not be prepared';
    setLoading(0, 'Dataset preparation failed.', 'Initial dataset loading will take time. You can close this message and try again.');
    close.hidden = false;
    throw error;
  }
}

async function prepareAndValidate(dataset, button) {
  button.disabled = true;
  try {
    const result = await prepareDataset(dataset, 'validate');
    if (result.cancelled) return;
    if (!result.url) throw new Error('Validation tools did not return a page address.');
    window.location.assign(result.url);
  } catch (error) { notice(error.message, 'error'); button.disabled = false; }
}

async function prepareAndView(dataset, button) {
  button.disabled = true;
  try {
    const result = await prepareDataset(dataset, 'viewer');
    if (result.cancelled) return;
    closeLoading();
    window.dispatchEvent(new CustomEvent('dataset:open', { detail: dataset.dataset_id }));
  } catch (error) { notice(error.message, 'error'); button.disabled = false; }
  finally { button.disabled = false; }
}

function resetDatasetLoadingAfterReturn() {
  preparationGeneration += 1;
  cancelPreparationDelay?.(); cancelPreparationDelay = null;
  closeLoading();
  $('dataset-loading-close').hidden = true;
  document.querySelectorAll('.dataset-card .item-actions button').forEach(button => { button.disabled = false; });
}

async function deleteDataset(dataset, button) {
  const proceed = window.confirm(`Permanently delete dataset “${dataset.name || dataset.dataset_id}”? This cannot be undone.`);
  if (!proceed) return;
  button.disabled = true;
  try {
    const result = await api.deleteDataset(dataset.path || dataset.dataset_id);
    notice(result.message || 'Dataset deleted.', 'success');
    await loadDatasets();
  } catch (error) { notice(error.message, 'error'); button.disabled = false; }
}

export function initDatasets() {
  $('refresh-datasets').addEventListener('click', loadDatasets);
  $('dataset-loading-close').addEventListener('click', closeLoading);
  window.addEventListener('pageshow', resetDatasetLoadingAfterReturn);
  window.addEventListener('pagehide', resetDatasetLoadingAfterReturn);
  window.addEventListener('catalog:refresh-datasets', loadDatasets);
}
