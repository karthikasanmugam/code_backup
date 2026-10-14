import { api } from './api.js';
import { state } from './state.js';
import { initTraining, loadPolicies } from './training.js';
import { initDatasets, loadDatasets } from './datasets.js';
import { initCheckpoints, loadCheckpoints } from './checkpoints.js';
import { initInference, updateInferenceStatus } from './inference.js';
import { initEvaluation } from './evaluation.js';
import { initDatasetViewer } from './dataset-viewer.js';

const pageTitles = {
  training: 'Training workspace', inference: 'Inference & bridge', datasets: 'Datasets',
  checkpoints: 'Checkpoints', evaluation: 'Model evaluation',
};
let toastTimer = null;
let refreshBusy = false;

function showNotice(message, kind = 'info') {
  const toast = document.getElementById('toast');
  toast.textContent = message;
  toast.className = `toast ${kind === 'error' ? 'error' : kind === 'success' ? '' : 'info'}`;
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, 5200);
}

function navigate(tab) {
  if (!pageTitles[tab]) return;
  document.querySelectorAll('.nav-link').forEach(button => {
    const active = button.dataset.tab === tab;
    button.classList.toggle('active', active);
    button.setAttribute('aria-current', active ? 'page' : 'false');
  });
  document.querySelectorAll('.page-view').forEach(page => {
    const active = page.dataset.page === tab;
    page.classList.toggle('active', active);
    page.hidden = !active;
  });
  document.getElementById('page-title').textContent = pageTitles[tab];
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('mobile-menu').setAttribute('aria-expanded', 'false');
  if (tab === 'training' || tab === 'datasets') window.dispatchEvent(new Event('catalog:refresh-datasets'));
  if (tab === 'inference' || tab === 'checkpoints') window.dispatchEvent(new Event('catalog:refresh-checkpoints'));
  if (tab === 'evaluation') {
    window.dispatchEvent(new Event('catalog:refresh-datasets'));
    window.dispatchEvent(new Event('catalog:refresh-checkpoints'));
  }
}

function updateResource(value, valueId, meterId) {
  const number = Number(value);
  const available = value !== null && value !== undefined && Number.isFinite(number);
  document.getElementById(valueId).textContent = available ? `${Math.round(number)}%` : '—';
  document.getElementById(meterId).style.height = available ? `${Math.max(0, Math.min(number, 100))}%` : '0%';
}

function setBackendConnection(label, dotClass) {
  const dot = document.getElementById('header-status-dot');
  if (dot) {
    dot.classList.remove('online', 'busy', 'offline');
    dot.classList.add(dotClass);
  }
  const status = document.getElementById('header-status');
  if (status) status.textContent = label;
  const footer = document.getElementById('footer-backend-state');
  if (footer) footer.textContent = dotClass === 'offline' ? 'Backend unavailable' : 'Backend connected';
}

function applySystemStatus(status) {
  state.system = status;
  const system = status?.system || {};
  updateResource(system.cpu_percent, 'cpu-value', 'cpu-meter');
  updateResource(system.memory_percent, 'memory-value', 'memory-meter');
  updateResource(system.disk_percent, 'disk-value', 'disk-meter');
  updateResource(system.gpu_percent, 'gpu-value', 'gpu-meter');
  updateResource(system.vram_percent, 'vram-value', 'vram-meter');
  const running = Boolean(status?.training?.active || status?.inference?.server_running || status?.inference?.bridge_running);
  const dotClass = running ? 'busy' : 'online';
  setBackendConnection(running ? 'Backend connected · process active' : 'Backend connected', dotClass);
  window.dispatchEvent(new CustomEvent('system:status', { detail: status }));
}

async function refreshStatus() {
  if (refreshBusy) return;
  refreshBusy = true;
  try {
    const [systemResult, trainingResult] = await Promise.allSettled([api.status(), api.trainStatus()]);
    if (trainingResult.status === 'fulfilled') {
      state.training = trainingResult.value;
      window.dispatchEvent(new CustomEvent('training:status', { detail: trainingResult.value }));
    }
    if (systemResult.status === 'fulfilled') applySystemStatus(systemResult.value);
    else setBackendConnection('Backend unavailable', 'offline');
  } finally {
    refreshBusy = false;
  }
}

function initTheme() {
  let preference = '';
  try { preference = localStorage.getItem('daksha.theme') || ''; } catch (_) { /* storage may be disabled */ }
  if (preference === 'dark') document.body.dataset.theme = 'dark';
  const button = document.getElementById('theme-toggle');
  const themeColor = document.querySelector('meta[name="theme-color"]');
  const renderIcon = () => {
    const dark = document.body.dataset.theme === 'dark';
    button.innerHTML = dark
      ? '<svg class="theme-icon" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.93 4.93l1.42 1.42m11.3 11.3 1.42 1.42M2 12h2m16 0h2M4.93 19.07l1.42-1.42m11.3-11.3 1.42-1.42"/></svg>'
      : '<svg class="theme-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20.2 15.4A8.5 8.5 0 0 1 8.6 3.8 8.7 8.7 0 1 0 20.2 15.4Z"/></svg>';
    const label = dark ? 'Switch to light theme' : 'Switch to dark theme';
    button.setAttribute('aria-label', label);
    button.setAttribute('title', label);
    button.setAttribute('aria-pressed', String(dark));
    if (themeColor) themeColor.content = dark ? '#090b0d' : '#0a2442';
  };
  renderIcon();
  button.addEventListener('click', () => {
    const next = document.body.dataset.theme === 'dark' ? 'light' : 'dark';
    if (next === 'light') delete document.body.dataset.theme;
    else document.body.dataset.theme = next;
    try { localStorage.setItem('daksha.theme', next); } catch (_) { /* preference stays in memory */ }
    renderIcon();
  });
}

function initNavigation() {
  document.querySelectorAll('.nav-link').forEach(button => button.addEventListener('click', () => navigate(button.dataset.tab)));
  document.getElementById('mobile-menu').addEventListener('click', event => {
    const sidebar = document.getElementById('sidebar');
    const open = sidebar.classList.toggle('open');
    event.currentTarget.setAttribute('aria-expanded', String(open));
  });
  window.addEventListener('app:navigate', event => navigate(event.detail));
}

function init() {
  initNavigation(); initTheme();
  const initialTab = new URLSearchParams(window.location.search).get('tab');
  if (pageTitles[initialTab]) navigate(initialTab);
  initTraining(); initDatasets(); initCheckpoints(); initInference(); initEvaluation(); initDatasetViewer();
  window.addEventListener('app:notice', event => {
    const detail = event.detail || {};
    showNotice(detail.message || 'Request completed.', detail.kind || 'info');
  });

  loadDatasets();
  loadCheckpoints();
  api.policies().then(loadPolicies).catch(error => {
    loadPolicies({}); showNotice(`Could not load policy catalog: ${error.message}`, 'error');
  });
  refreshStatus();
  window.setInterval(refreshStatus, 2500);
}

init();
