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
let loadingDataset = false;
let startingJob = false;
let stoppingJob = false;

function exactCheckpoints(model) {
  const records = state.checkpoints.filter(run => !model || run.model === model);
  return records.flatMap(run => (run.checkpoint_options || []).map(option => ({
    path: option.path,
    label: `${String(run.model || 'policy').toUpperCase()} · ${run.task || run.name} · ${option.label || option.path}`,
  })));
}

function refreshEvaluationOptions() {
  const datasetSelect = $('mv-dataset');
  const checkpointSelect = $('mv-checkpoint');
  const priorDataset = datasetSelect.value;
  const priorCheckpoint = checkpointSelect.value;
  datasetSelect.replaceChildren(new Option(state.datasets.length ? 'Select dataset' : 'No datasets available', ''));
  state.datasets.forEach(item => datasetSelect.add(new Option(item.name || item.dataset_id, item.dataset_id)));
  if (state.datasets.some(item => item.dataset_id === priorDataset)) datasetSelect.value = priorDataset;
  const options = exactCheckpoints($('mv-model-type').value);
  checkpointSelect.replaceChildren(new Option(options.length ? 'Select checkpoint step' : 'No saved steps for this policy', ''));
  options.forEach(item => checkpointSelect.add(new Option(item.label, item.path)));
  if (options.some(item => item.path === priorCheckpoint)) checkpointSelect.value = priorCheckpoint;
  resetLoadedDataset();
}

function setError(message) {
  const target = $('mv-error'); target.textContent = message || ''; target.hidden = !message;
}

function setProgress(value, message) {
  const amount = Math.max(0, Math.min(100, Number(value || 0)));
  $('mv-progress-fill').style.width = `${amount}%`;
  $('mv-progress-percent').textContent = message && amount === 0 ? '' : `${amount.toFixed(0)}%`;
  $('mv-progress-text').textContent = message || 'Working…';
}

function resetLoadedDataset() {
  state.evaluation.loaded = null;
  const episode = $('mv-episode'); episode.replaceChildren(new Option('Load dataset first', '')); episode.disabled = true;
  $('mv-eval-btn').disabled = true; $('mv-dataset-meta').textContent = '';
  $('mv-episode-note').textContent = '';
  clearResult();
  updateScope();
}

function updateScope() {
  const scope = $('mv-scope').value;
  $('mv-episode').disabled = !state.evaluation.loaded || scope === 'dataset';
  $('mv-episode-note').textContent = scope === 'dataset' && state.evaluation.loaded ? 'All listed held-out episodes will be evaluated.' : '';
}

async function loadHeldOutDataset() {
  if (loadingDataset) return;
  const datasetId = $('mv-dataset').value;
  const checkpoint = $('mv-checkpoint').value;
  if (!datasetId || !checkpoint) { setError('Select a dataset and checkpoint first.'); return; }
  loadingDataset = true; setError(''); $('mv-load-btn').disabled = true; $('mv-eval-btn').disabled = true;
  $('mv-dataset-meta').textContent = 'Loading held-out episode metadata…';
  try {
    const dataset = state.datasets.find(item => item.dataset_id === datasetId);
    const data = await api.datasetInfoForEvaluation({ dataset: dataset?.path || datasetId, checkpoint });
    state.evaluation.loaded = { ...data, dataset: dataset?.path || datasetId, checkpoint };
    const episode = $('mv-episode'); episode.replaceChildren();
    (data.episodes || []).forEach(item => {
      const task = item.task ? ` · ${item.task}` : '';
      episode.add(new Option(`Episode ${item.episode_index} · ${Number(item.length || 0).toLocaleString()} frames${task}`, item.episode_index));
    });
    $('mv-eval-btn').disabled = !(data.episodes || []).length || Boolean(state.evaluation.jobId);
    const splitState = data.held_out_during_training ? `held out during training (eval split ${Number(data.eval_split).toFixed(2)})` : `split could not be confirmed (eval split ${Number(data.eval_split).toFixed(2)})`;
    $('mv-dataset-meta').textContent = `${data.episodes?.length || 0} held-out episode${data.episodes?.length === 1 ? '' : 's'} · ${Number(data.frames || 0).toLocaleString()} frames · ${data.total_episodes || '—'} total · ${data.fps || '—'} FPS · ${data.cameras?.length || 0} cameras · ${splitState}`;
    updateScope(); setError('');
    if (!data.episodes?.length) setError('This dataset and checkpoint have no held-out evaluation episodes.');
  } catch (error) {
    $('mv-dataset-meta').textContent = ''; setError(error.message);
  } finally { loadingDataset = false; $('mv-load-btn').disabled = Boolean(state.evaluation.jobId); }
}

async function startEvaluation() {
  if (startingJob || !state.evaluation.loaded) return;
  const loaded = state.evaluation.loaded;
  const scope = $('mv-scope').value;
  const episode = $('mv-episode').value;
  if (scope === 'episode' && episode === '') { setError('Select a held-out test episode.'); return; }
  const stride = Math.max(1, Math.trunc(Number($('mv-stride').value) || 1));
  const maxText = $('mv-max-frames').value.trim();
  const seed = Math.max(0, Math.trunc(Number($('mv-seed').value) || 0));
  startingJob = true; setError(''); clearResult(); setProgress(0, 'Queuing evaluation…'); updateEvaluationButtons();
  try {
    const result = await api.startEvaluation({
      model_type: $('mv-model-type').value,
      inference_profile: 'checkpoint',
      dataset: loaded.dataset,
      checkpoint: loaded.checkpoint,
      scope,
      ...(scope === 'episode' ? { episode: Number(episode) } : {}),
      device: $('mv-device').value,
      stride,
      max_frames: maxText ? Math.max(1, Math.trunc(Number(maxText))) : null,
      seed,
    });
    state.evaluation.jobId = result.job_id;
    setProgress(0, 'Queued'); startPolling();
  } catch (error) { setError(error.message); setProgress(0, 'Evaluation not started'); }
  finally { startingJob = false; updateEvaluationButtons(); }
}

function updateEvaluationButtons() {
  const active = Boolean(state.evaluation.jobId) || startingJob || stoppingJob;
  $('mv-load-btn').disabled = active || loadingDataset;
  $('mv-eval-btn').disabled = active || startingJob || !state.evaluation.loaded || !state.evaluation.loaded.episodes?.length;
  $('mv-stop-btn').disabled = !state.evaluation.jobId || stoppingJob;
  $('mv-reset-btn').disabled = active;
  ['mv-dataset', 'mv-model-type', 'mv-checkpoint', 'mv-scope', 'mv-episode'].forEach(id => { $(id).disabled = active; });
}

function startPolling() {
  clearTimeout(state.evaluation.pollTimer);
  state.evaluation.pollTimer = setTimeout(pollJob, 800);
}

async function pollJob() {
  const jobId = state.evaluation.jobId;
  if (!jobId) return;
  try {
    const job = await api.evaluationStatus(jobId);
    const status = job.status || 'running';
    setProgress(job.progress || 0, job.message || status);
    if (status === 'complete') {
      state.evaluation.result = job.result;
      setProgress(100, 'Evaluation complete');
      renderResult(job.result || {}); finishJob(); return;
    }
    if (['error', 'cancelled', 'stopped'].includes(status)) {
      if (status === 'error') setError(job.message || 'Evaluation failed.');
      finishJob(); return;
    }
    state.evaluation.pollTimer = setTimeout(pollJob, 900);
  } catch (error) {
    setError(error.message); finishJob();
  }
}

function finishJob() {
  clearTimeout(state.evaluation.pollTimer); state.evaluation.pollTimer = null;
  state.evaluation.jobId = ''; stoppingJob = false; startingJob = false;
  updateEvaluationButtons();
}

async function stopEvaluation() {
  if (!state.evaluation.jobId || stoppingJob) return;
  stoppingJob = true; updateEvaluationButtons(); setProgress(0, 'Stopping evaluation…');
  try { await api.stopEvaluation(state.evaluation.jobId); startPolling(); }
  catch (error) { setError(error.message); stoppingJob = false; updateEvaluationButtons(); }
}

function clearResult() {
  state.evaluation.result = null;
  $('mv-result-state').textContent = 'No completed result';
  $('mv-result-details').hidden = true;
  [$('mv-chart'), $('mv-joint-chart')].forEach(root => {
    root.querySelectorAll('.js-plotly-plot').forEach(plot => window.Plotly?.purge(plot));
    root.replaceChildren();
  });
  $('mv-chart-note').textContent = '';
  ['mv-m-episode', 'mv-m-episodes', 'mv-m-frames', 'mv-m-mae', 'mv-m-rmse', 'mv-m-max-rmse', 'mv-m-p50', 'mv-m-aps'].forEach(id => { $(id).textContent = '—'; });
}

function renderActionComparison(result) {
  const root = $('mv-chart'); root.replaceChildren();
  const actual = result.actual, predicted = result.predicted;
  const observedState = result.observed_state;
  const hasObservedState = Array.isArray(observedState) && observedState.length === actual?.length;
  const dim = Number(result.action_dim || result.joint_names?.length || actual?.[0]?.length || 0);
  if (!Array.isArray(actual) || !actual.length || !Array.isArray(predicted) || predicted.length !== actual.length || !dim) {
    root.append(el('p', 'muted', 'Action sequence data was not included in the completed result. Summary metrics are shown above.'));
    return;
  }
  if (!window.Plotly) {
    root.append(el('p', 'muted', 'Interactive charts could not load. Check the connection to cdn.plot.ly and reload the page.'));
    return;
  }

  const xValues = result.global_indices || result.frame_indices || actual.map((_, index) => index);
  const stride = Math.max(1, Math.ceil(actual.length / 1000));
  const episodeIds = Array.isArray(result.episode_indices) ? result.episode_indices : [];
  for (let joint = 0; joint < dim; joint += 1) {
    const x = [], recordedValues = [], predictedValues = [], stateValues = [];
    let previousEpisode = episodeIds[0] ?? null;
    for (let i = 0; i < actual.length; i += stride) {
      const a = Number(actual[i]?.[joint]), p = Number(predicted[i]?.[joint]);
      if (!Number.isFinite(a) || !Number.isFinite(p)) continue;
      const episode = episodeIds[i] ?? null;
      if (i > 0 && episode !== previousEpisode) {
        x.push(null); recordedValues.push(null); predictedValues.push(null); stateValues.push(null);
      }
      x.push(Number(xValues[i] ?? i));
      recordedValues.push(a);
      predictedValues.push(p);
      const state = Number(observedState?.[i]?.[joint]);
      stateValues.push(hasObservedState && Number.isFinite(state) ? state : null);
      previousEpisode = episode;
    }
    if (!recordedValues.some(Number.isFinite)) continue;
    const name = result.joint_names?.[joint] || `Action ${joint + 1}`;
    const graph = el('div', 'eval-joint-plot');
    graph.setAttribute('role', 'img');
    graph.setAttribute('aria-label', `Recorded action, predicted action, and observed state for ${name}`);
    root.append(graph);
    const series = [
      { x, y: recordedValues, type: 'scatter', mode: 'lines', name: 'Recorded action', line: { color: '#218d86', width: 2 }, connectgaps: false },
      { x, y: predictedValues, type: 'scatter', mode: 'lines', name: 'Predicted action', line: { color: '#e0944d', width: 2 }, connectgaps: false },
      { x, y: stateValues, type: 'scatter', mode: 'lines', name: 'Observed state', line: { color: '#4f80a6', width: 2 }, connectgaps: false },
    ];
    document.querySelectorAll('[data-chart-series]').forEach(button => {
      series[Number(button.dataset.chartSeries)].visible = button.getAttribute('aria-pressed') === 'true';
    });
    const mae = result.metrics?.mae?.[joint], rmse = result.metrics?.rmse?.[joint];
    window.Plotly.newPlot(graph, series, {
      title: { text: `${String(joint + 1).padStart(2, '0')} · ${name} · MAE ${Number.isFinite(Number(mae)) ? Number(mae).toFixed(4) : '—'} · RMSE ${Number.isFinite(Number(rmse)) ? Number(rmse).toFixed(4) : '—'}`, font: { size: 12, color: '#476477' } },
      autosize: true, height: 290, margin: { l: 68, r: 18, t: 42, b: 58 },
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: '#fcfdfd',
      font: { family: 'DM Sans, sans-serif', size: 11, color: '#627e8e' },
      hoverlabel: { bgcolor: '#ffffff', bordercolor: '#c6d2d9', font: { color: '#1d3542', size: 12 }, align: 'left' },
      hovermode: 'x unified', showlegend: false,
      xaxis: { title: { text: 'Dataset frame index' }, type: 'linear', tickformat: ',d', automargin: true, showgrid: true, zeroline: false },
      yaxis: { title: { text: 'Value (dataset units)' }, tickformat: '.4f', automargin: true, showgrid: true, zeroline: false },
      dragmode: false,
    }, { responsive: true, displayModeBar: false, displaylogo: false, scrollZoom: false, doubleClick: false });
  }
}

function renderJointErrors(result) {
  const root = $('mv-joint-chart'); root.replaceChildren();
  const values = result.metrics?.rmse || [];
  const names = result.joint_names || [];
  if (!values.length) { root.append(el('p', 'muted', 'Per-joint error values are unavailable.')); return; }
  if (!window.Plotly) return;
  window.Plotly.newPlot(root, [{
    x: values.map(Number), y: values.map((_, index) => names[index] || `Action ${index + 1}`),
    type: 'bar', orientation: 'h', marker: { color: '#e6a45d' },
    hovertemplate: '%{y}<br>RMSE: %{x:.5f}<extra></extra>',
  }], {
    title: { text: 'Per-channel RMSE', font: { size: 12, color: '#476477' } },
    autosize: true, height: Math.max(240, values.length * 28 + 90),
    margin: { l: 150, r: 24, t: 42, b: 54 },
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: '#fcfdfd',
    font: { family: 'DM Sans, sans-serif', size: 10, color: '#627e8e' },
    hoverlabel: { bgcolor: '#ffffff', bordercolor: '#c6d2d9', font: { color: '#1d3542', size: 12 }, align: 'left' },
    xaxis: { title: { text: 'RMSE (dataset units)' }, tickformat: '.5f', automargin: true, showgrid: true, zeroline: true },
    yaxis: { title: { text: 'Action channel' }, automargin: true, showgrid: false },
    dragmode: false,
  }, { responsive: true, displayModeBar: false, displaylogo: false, scrollZoom: false, doubleClick: false });
}

function renderResult(result) {
  clearResult();
  state.evaluation.result = result;
  const metrics = result.metrics || {};
  const val = (value, digits = 4) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';
  $('mv-m-episode').textContent = result.scope === 'dataset' ? 'All' : String(result.episode_index ?? '—');
  $('mv-m-episodes').textContent = String(result.episode_count ?? '—');
  $('mv-m-frames').textContent = Number.isFinite(Number(result.evaluated_frames)) ? Number(result.evaluated_frames).toLocaleString() : '—';
  $('mv-m-mae').textContent = val(metrics.mean_mae, 5);
  $('mv-m-rmse').textContent = val(metrics.mean_rmse, 5);
  $('mv-m-max-rmse').textContent = val(metrics.max_joint_rmse, 5);
  $('mv-m-p50').textContent = val(result.latency?.p50_s, 4);
  $('mv-m-aps').textContent = val(result.latency?.actions_per_second, 2);
  $('mv-result-state').textContent = `${String(result.model_type || 'Policy').toUpperCase()} · complete`;
  $('mv-result-details').hidden = false;
  renderActionComparison(result); renderJointErrors(result);
  $('mv-chart-note').textContent = `${result.scope === 'dataset' ? 'Held-out test split' : `Episode ${result.episode_index}`} · ${result.task || 'Multiple tasks'} · ${result.model_type || 'policy'} · ${result.evaluated_frames_per_second ? `${Number(result.evaluated_frames_per_second).toFixed(2)} evaluated frames/s` : 'runtime unavailable'}`;
}

function resetForm() {
  if (state.evaluation.jobId) return;
  $('mv-model-type').value = 'act'; $('mv-scope').value = 'dataset'; $('mv-device').value = 'cuda';
  $('mv-stride').value = '1'; $('mv-max-frames').value = ''; $('mv-seed').value = '0';
  $('mv-dataset').value = ''; $('mv-checkpoint').value = '';
  setError(''); setProgress(0, 'No evaluation running'); refreshEvaluationOptions();
}

async function restoreActiveJob() {
  try {
    const job = await api.activeEvaluation();
    if (job?.job_id && ['queued', 'loading', 'running', 'stopping'].includes(job.status)) {
      state.evaluation.jobId = job.job_id;
      setProgress(job.progress || 0, job.message || job.status);
      updateEvaluationButtons(); startPolling();
    }
  } catch (error) { setError(`Could not restore active evaluation: ${error.message}`); }
}

export function initEvaluation() {
  const seriesControls = $('mv-chart-series-controls');
  if (seriesControls) {
    seriesControls.addEventListener('click', event => {
      const button = event.target.closest('[data-chart-series]');
      if (!button) return;
      const visible = button.getAttribute('aria-pressed') !== 'true';
      const traceIndex = Number(button.dataset.chartSeries);
      button.setAttribute('aria-pressed', String(visible));
      button.classList.toggle('is-active', visible);
      document.querySelectorAll('.eval-joint-plot.js-plotly-plot').forEach(plot => {
        window.Plotly?.restyle(plot, { visible }, [traceIndex]);
      });
    });
  }
  $('refresh-evaluation-options').addEventListener('click', () => {
    window.dispatchEvent(new Event('catalog:refresh-datasets'));
    window.dispatchEvent(new Event('catalog:refresh-checkpoints'));
  });
  $('mv-model-type').addEventListener('change', () => { refreshEvaluationOptions(); });
  $('mv-dataset').addEventListener('change', resetLoadedDataset);
  $('mv-checkpoint').addEventListener('change', resetLoadedDataset);
  $('mv-scope').addEventListener('change', updateScope);
  $('mv-load-btn').addEventListener('click', loadHeldOutDataset);
  $('mv-eval-btn').addEventListener('click', startEvaluation);
  $('mv-stop-btn').addEventListener('click', stopEvaluation);
  $('mv-reset-btn').addEventListener('click', resetForm);
  window.addEventListener('catalog:datasets', refreshEvaluationOptions);
  window.addEventListener('catalog:checkpoints', refreshEvaluationOptions);
  restoreActiveJob();
}
