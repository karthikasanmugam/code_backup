import { api } from './api.js';
import { state } from './state.js';

const $ = id => document.getElementById(id);
const el = (tag, className = '', text = '') => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== '') node.textContent = text;
  return node;
};
const ns = 'http://www.w3.org/2000/svg';
const MAX_CHART_POINTS = 700;
let episodeToken = 0;
let episodesRequestToken = 0;
let syncingMedia = false;
let activeVideos = [];
let chartRefs = [];
let episodeDuration = 0;
let currentVideoDuration = 0;
let actionRequestController = null;
let episodeChangeTimer = null;
let episodesRequestController = null;

function notify(message, kind = 'info') {
  window.dispatchEvent(new CustomEvent('app:notice', { detail: { message, kind } }));
}

function setViewerStatus(message) { $('viewer-media-state').textContent = message; }

function metaCard(label, value, detail = '') {
  const card = el('div', 'viewer-meta-card');
  card.append(el('small', '', label), el('strong', '', value === null || value === undefined || value === '' ? '—' : String(value)));
  if (detail) card.append(el('span', '', detail));
  return card;
}

function renderViewerMetadata(metadata) {
  $('viewer-title').textContent = metadata.dataset_id || state.viewer.datasetId;
  const suffix = [metadata.version, metadata.layout, metadata.robot_type].filter(Boolean).join(' · ');
  $('viewer-subtitle').textContent = suffix || 'Dataset capabilities and recorded episodes';
  const roots = $('viewer-meta'); roots.replaceChildren();
  roots.append(
    metaCard('Episodes', Number(metadata.total_episodes || 0).toLocaleString(), 'from dataset metadata'),
    metaCard('Frames', Number(metadata.total_frames || 0).toLocaleString(), 'recorded frames'),
    metaCard('Capture rate', metadata.fps ? `${metadata.fps} FPS` : '—', metadata.layout || 'frame rate'),
    metaCard('Actions', `${metadata.action_dim || 0} channels`, (metadata.action_names || []).slice(0, 4).join(' · ')),
  );
  const tasks = Array.isArray(metadata.tasks) ? metadata.tasks : [];
  const select = $('viewer-task');
  select.replaceChildren(new Option('All recorded tasks', ''));
  tasks.forEach(task => {
    const prompt = task.prompt || task.task || '';
    const amount = task.count ? ` (${task.count})` : '';
    select.add(new Option(`${prompt || `Task ${task.task_index ?? ''}`}${amount}`, prompt));
  });
  state.viewer.total = Number(metadata.total_episodes || 0);
}

function pauseAndClearMedia() {
  activeVideos.forEach(record => {
    try { record.video.pause(); record.video.removeAttribute('src'); record.video.load(); } catch (_) { /* detached video */ }
  });
  activeVideos = [];
  $('viewer-play').disabled = true; $('viewer-pause').disabled = true; $('viewer-stop').disabled = true;
  $('viewer-seek').disabled = true; $('viewer-seek').value = '0'; $('viewer-seek').max = '0';
  $('viewer-clock').textContent = '00:00 / 00:00'; $('viewer-seek-time').textContent = '00:00';
}

function resetActions(message = 'Action charts load when an episode is selected.') {
  state.viewer.actions = null; state.viewer.actionPage = 0; chartRefs = [];
  $('action-chart-grid').replaceChildren(el('div', 'empty-state', message));
  $('action-dimension-label').textContent = 'No action data loaded';
  $('action-chart-title').textContent = 'Action channels'; $('action-chart-range').textContent = '';
  $('action-page-label').textContent = '—';
  $('action-page-prev').disabled = true; $('action-page-next').disabled = true;
  $('action-current').textContent = 'Frame — · Time —';
}

function resetEpisodeView(message = 'Choose a dataset episode to load its available camera recordings.') {
  actionRequestController?.abort(); actionRequestController = null;
  state.viewer.episode = null; state.viewer.videos = [];
  pauseAndClearMedia(); resetActions();
  $('viewer-prompt').textContent = 'Select an episode to view its task.';
  $('viewer-video-grid').replaceChildren(el('div', 'empty-state', message));
  setViewerStatus('Waiting for episode');
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '00:00';
  const whole = Math.floor(seconds);
  const minutes = Math.floor(whole / 60);
  return `${String(minutes).padStart(2, '0')}:${String(whole % 60).padStart(2, '0')}`;
}

function sortCameraRecords(videos) {
  const priority = key => {
    const value = key.toLowerCase();
    if (value.includes('wrist_left') || value.includes('left_wrist')) return 0;
    if (value.includes('world') || value.includes('center')) return 1;
    if (value.includes('wrist_right') || value.includes('right_wrist')) return 2;
    return 3;
  };
  return [...(videos || [])].sort((a, b) => priority(a.camera_key) - priority(b.camera_key));
}

function recordStart(record) { return Number(record.start_time ?? record.start_s ?? 0) || 0; }
function recordEnd(record) { const end = Number(record.end_time ?? record.end_s); return Number.isFinite(end) ? end : recordStart(record) + episodeDuration; }
function recordPlayableEnd(record) {
  const mediaEnd = Number.isFinite(record.video.duration) ? record.video.duration : Infinity;
  return Math.min(recordEnd(record), mediaEnd);
}
function relTime(video) {
  const record = activeVideos.find(item => item.video === video);
  return Math.max(0, video.currentTime - (record ? recordStart(record) : 0));
}
function masterRecord() {
  return activeVideos.find(item => /world|center/i.test(item.camera_key)) || activeVideos[0] || null;
}

function setMediaButtons(ready) {
  $('viewer-play').disabled = !ready;
  $('viewer-pause').disabled = !ready;
  $('viewer-stop').disabled = !ready;
  $('viewer-seek').disabled = !ready;
}

function updateClock(relative) {
  const total = currentVideoDuration || episodeDuration;
  const safe = Math.max(0, Math.min(total || 0, relative || 0));
  $('viewer-clock').textContent = `${formatTime(safe)} / ${formatTime(total)}`;
  $('viewer-seek-time').textContent = formatTime(safe);
  $('viewer-seek').value = String(safe);
  updateActionCursor(safe);
}

function seekAll(relative) {
  const safe = Math.max(0, Math.min(currentVideoDuration || episodeDuration || 0, relative || 0));
  syncingMedia = true;
  activeVideos.forEach(record => {
    if (record.video.readyState >= 1) {
      const end = recordPlayableEnd(record);
      record.video.currentTime = Math.min(recordStart(record) + safe, Math.max(recordStart(record), end - .02));
    }
  });
  syncingMedia = false;
  updateClock(safe);
}

function syncFromVideo(source, seek = false) {
  if (syncingMedia) return;
  const sourceRecord = activeVideos.find(item => item.video === source);
  if (!sourceRecord) return;
  const relative = Math.max(0, source.currentTime - recordStart(sourceRecord));
  syncingMedia = true;
  activeVideos.forEach(record => {
    if (record.video === source || record.video.readyState < 1) return;
    const target = Math.min(recordStart(record) + relative, Math.max(recordStart(record), recordPlayableEnd(record) - .02));
    if (seek || Math.abs(record.video.currentTime - target) > .12) record.video.currentTime = target;
  });
  syncingMedia = false;
  updateClock(relative);
  const duration = currentVideoDuration || episodeDuration;
  if (duration > 0 && relative >= duration - .06) {
    if ($('viewer-loop').checked) { seekAll(0); activeVideos.forEach(record => record.video.play().catch(() => {})); }
    else { activeVideos.forEach(record => record.video.pause()); updateClock(duration); }
  }
}

function renderVideoGrid(episode) {
  const root = $('viewer-video-grid'); root.replaceChildren(); activeVideos = [];
  const records = sortCameraRecords(episode.videos || []);
  if (!records.length) {
    root.append(el('div', 'empty-state', 'No camera views are advertised for this episode.'));
    setViewerStatus('No camera media'); setMediaButtons(false); return;
  }
  let availableCount = 0;
  records.forEach(record => {
    const card = el('article', 'viewer-video-card');
    const title = el('h4', '', record.label || record.camera_key);
    card.append(title);
    if (!record.available || !record.url) {
      card.append(el('div', 'unavailable', record.error || 'Video unavailable for this camera and episode.'), el('small', '', record.camera_key));
    } else {
      const video = document.createElement('video');
      video.playsInline = true; video.preload = 'metadata'; video.setAttribute('aria-label', `${record.label || record.camera_key} recording`);
      video.src = record.url;
      const sourceRecord = { video, camera_key: record.camera_key, start_time: record.start_time, end_time: record.end_time };
      activeVideos.push(sourceRecord); availableCount += 1;
      video.addEventListener('loadedmetadata', () => {
        video.currentTime = Math.min(recordStart(sourceRecord), Math.max(0, video.duration - .02));
        currentVideoDuration = Math.min(episodeDuration || Infinity, ...activeVideos.map(item => Math.max(0, recordEnd(item) - recordStart(item))).filter(Number.isFinite));
        if (!Number.isFinite(currentVideoDuration)) currentVideoDuration = episodeDuration;
        $('viewer-seek').max = String(currentVideoDuration || 0);
        setMediaButtons(activeVideos.some(item => item.video.readyState >= 1));
        setViewerStatus(`${availableCount} camera view${availableCount === 1 ? '' : 's'} ready`);
      });
      video.addEventListener('error', () => {
        const notice = el('div', 'unavailable', 'Browser could not load this recording.');
        video.replaceWith(notice);
        sourceRecord.failed = true; activeVideos = activeVideos.filter(item => item !== sourceRecord);
        if (!activeVideos.length) { setMediaButtons(false); setViewerStatus('Video unavailable'); }
      });
      // The hidden per-camera players are driven by the shared controls. Let only
      // the world camera clock update the other streams and charts; listening to
      // seeking on every synchronized player can create a seek-event feedback loop.
      video.addEventListener('timeupdate', () => {
        if (video === masterRecord()?.video) syncFromVideo(video);
      });
      card.append(video, el('small', '', record.camera_key));
    }
    root.append(card);
  });
  if (!availableCount) { setViewerStatus('Video unavailable'); setMediaButtons(false); }
  else setViewerStatus('Loading camera media…');
}

function nearestActionSample(relative) {
  const samples = state.viewer.actions?.samples || [];
  if (!samples.length) return null;
  let low = 0, high = samples.length - 1;
  while (low < high) {
    const mid = (low + high) >> 1;
    if (Number(samples[mid].episode_time || 0) < relative) low = mid + 1;
    else high = mid;
  }
  if (low > 0 && Math.abs(Number(samples[low - 1].episode_time) - relative) < Math.abs(Number(samples[low].episode_time) - relative)) low -= 1;
  return samples[low];
}

function updateActionCursor(relative) {
  if (!chartRefs.length) return;
  const sample = nearestActionSample(relative);
  chartRefs.forEach(ref => {
    const progress = ref.timeMax > ref.timeMin
      ? Math.max(0, Math.min(1, (relative - ref.timeMin) / (ref.timeMax - ref.timeMin)))
      : 0;
    ref.cursor.setAttribute('x1', String(ref.left + progress * ref.plotWidth));
    ref.cursor.setAttribute('x2', String(ref.left + progress * ref.plotWidth));
    const value = sample?.action?.[ref.channel];
    ref.value.textContent = Number.isFinite(Number(value)) ? Number(value).toFixed(4) : '—';
  });
  $('action-current').textContent = sample
    ? `Frame ${sample.frame_index ?? '—'} · Time ${Number(relative).toFixed(2)} s · Nearest sample ${Number(sample.episode_time || 0).toFixed(2)} s`
    : `Time ${Number(relative).toFixed(2)} s · Action sample unavailable`;
}

function renderActionCharts(page = 0) {
  const payload = state.viewer.actions;
  const root = $('action-chart-grid'); root.replaceChildren(); chartRefs = [];
  const names = payload?.action_names || [];
  const samples = payload?.samples || [];
  const dim = names.length;
  if (!dim || !samples.length) {
    root.append(el('div', 'empty-state', 'This episode does not provide action samples.'));
    $('action-dimension-label').textContent = 'Actions unavailable';
    $('action-page-label').textContent = '—'; $('action-page-prev').disabled = true; $('action-page-next').disabled = true; return;
  }
  const pageCount = Math.ceil(dim / 8);
  state.viewer.actionPage = Math.max(0, Math.min(page, pageCount - 1));
  const start = state.viewer.actionPage * 8, end = Math.min(dim, start + 8);
  const group = names.slice(start, end);
  const groupText = group.map(name => String(name).toLowerCase());
  const groupTitle = groupText.every(name => name.includes('left')) ? 'Left arm'
    : groupText.every(name => name.includes('right')) ? 'Right arm'
      : `Channels ${start + 1}–${end}`;
  $('action-chart-title').textContent = groupTitle;
  $('action-chart-range').textContent = ` · ${start + 1}–${end} of ${dim}`;
  $('action-dimension-label').textContent = `${dim} action channel${dim === 1 ? '' : 's'} · ${samples.length.toLocaleString()} samples`;
  $('action-page-label').textContent = `${state.viewer.actionPage + 1} / ${pageCount}`;
  $('action-page-prev').disabled = state.viewer.actionPage === 0;
  $('action-page-next').disabled = state.viewer.actionPage >= pageCount - 1;

  for (let channel = start; channel < end; channel += 1) {
    const card = el('article', 'action-chart-card');
    card.append(el('h4', '', `${String(channel + 1).padStart(2, '0')} · ${names[channel]}`));
    const svg = document.createElementNS(ns, 'svg'); svg.setAttribute('viewBox', '0 0 210 80'); svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', `Action channel ${names[channel]}`);
    const left = 3, top = 6, plotWidth = 202, plotHeight = 60;
    [0, .5, 1].forEach(frac => {
      const line = document.createElementNS(ns, 'line'); line.setAttribute('x1', String(left)); line.setAttribute('x2', String(left + plotWidth)); line.setAttribute('y1', String(top + frac * plotHeight)); line.setAttribute('y2', String(top + frac * plotHeight)); line.setAttribute('class', 'chart-gridline'); svg.append(line);
    });
    let min = Infinity, max = -Infinity, timeMin = Infinity, timeMax = -Infinity;
    samples.forEach((item, index) => {
      const value = Number(item.action?.[channel]);
      const time = Number(item.episode_time ?? item.timestamp);
      if (Number.isFinite(value)) { min = Math.min(min, value); max = Math.max(max, value); }
      if (Number.isFinite(time)) { timeMin = Math.min(timeMin, time); timeMax = Math.max(timeMax, time); }
    });
    if (!Number.isFinite(min) || !Number.isFinite(max)) { min = 0; max = 0; }
    if (!Number.isFinite(timeMin) || !Number.isFinite(timeMax)) { timeMin = 0; timeMax = Math.max(samples.length - 1, 0); }
    const spread = max - min || Math.max(Math.abs(max) * .08, .001);
    const path = document.createElementNS(ns, 'path');
    const coords = [];
    const stride = Math.max(1, Math.ceil(samples.length / MAX_CHART_POINTS));
    const plotted = samples.filter((_, index) => index % stride === 0 || index === samples.length - 1);
    plotted.forEach((item, index) => {
      const value = Number(item.action?.[channel]); if (!Number.isFinite(value)) return;
      const time = Number(item.episode_time ?? item.timestamp);
      const progress = Number.isFinite(time) && timeMax > timeMin
        ? (time - timeMin) / (timeMax - timeMin)
        : index / Math.max(plotted.length - 1, 1);
      const px = left + progress * plotWidth;
      const py = top + (max - value) / spread * plotHeight;
      coords.push(`${coords.length ? 'L' : 'M'}${px.toFixed(1)},${py.toFixed(1)}`);
    });
    path.setAttribute('d', coords.join(' ')); path.setAttribute('class', 'chart-line'); svg.append(path);
    const axis = document.createElementNS(ns, 'line');
    axis.setAttribute('x1', String(left)); axis.setAttribute('x2', String(left + plotWidth));
    axis.setAttribute('y1', String(top + plotHeight)); axis.setAttribute('y2', String(top + plotHeight));
    axis.setAttribute('class', 'chart-gridline'); svg.append(axis);
    [[left, `${timeMin.toFixed(2)}s`, 'start'], [left + plotWidth, `${timeMax.toFixed(2)}s`, 'end']].forEach(([x, label, edge]) => {
      const tick = document.createElementNS(ns, 'text');
      tick.setAttribute('x', String(x)); tick.setAttribute('y', '77');
      tick.setAttribute('text-anchor', edge === 'start' ? 'start' : 'end');
      tick.setAttribute('class', 'chart-axis-label'); tick.textContent = label; svg.append(tick);
    });
    const cursor = document.createElementNS(ns, 'line'); cursor.setAttribute('y1', String(top)); cursor.setAttribute('y2', String(top + plotHeight)); cursor.setAttribute('class', 'chart-cursor'); svg.append(cursor);
    card.append(svg);
    const value = el('strong', '', '—'); card.append(value); root.append(card);
    chartRefs.push({ cursor, value, channel, left, plotWidth, timeMin, timeMax });
  }
  updateActionCursor(0);
}

async function loadEpisode(episodeIndex) {
  const episode = state.viewer.episodes.find(item => Number(item.episode_index) === Number(episodeIndex));
  if (!episode) return;
  actionRequestController?.abort();
  const controller = new AbortController();
  actionRequestController = controller;
  const token = ++episodeToken;
  state.viewer.episode = episode;
  pauseAndClearMedia(); resetActions('Loading selected episode action samples…');
  $('viewer-prompt').textContent = episode.task || episode.prompt || 'No task prompt recorded for this episode.';
  episodeDuration = Number(episode.duration_s || episode.duration || 0);
  currentVideoDuration = episodeDuration;
  renderVideoGrid(episode);
  setViewerStatus('Loading action samples…');
  try {
    const actions = await api.actions(state.viewer.datasetId, episode.episode_index, { signal: controller.signal });
    if (token !== episodeToken) return;
    state.viewer.actions = actions;
    renderActionCharts(0);
    setViewerStatus(activeVideos.length ? 'Loading camera media…' : 'No camera media');
  } catch (error) {
    if (token !== episodeToken || error.name === 'AbortError') return;
    resetActions(error.message); notify(error.message, 'error');
  } finally {
    if (actionRequestController === controller) actionRequestController = null;
  }
}

function updateEpisodeSelect(keepValue = '') {
  const select = $('viewer-episode');
  const entries = state.viewer.episodes;
  select.replaceChildren(new Option(entries.length ? 'Select episode' : 'No episodes found', ''));
  entries.forEach(item => {
    const label = `Episode ${item.episode_index} · ${Number(item.frame_count || 0).toLocaleString()} frames · ${formatTime(Number(item.duration_s || 0))}`;
    select.add(new Option(label, String(item.episode_index)));
  });
  if (keepValue && entries.some(item => String(item.episode_index) === String(keepValue))) select.value = String(keepValue);
  select.disabled = !entries.length;
  $('viewer-episode-count').textContent = `${state.viewer.total.toLocaleString()} total episodes`;
}

async function loadEpisodePage(reset = false) {
  if (!state.viewer.datasetId) return;
  episodesRequestController?.abort();
  const controller = new AbortController();
  episodesRequestController = controller;
  const requestToken = ++episodesRequestToken;
  const datasetId = state.viewer.datasetId;
  const task = state.viewer.task;
  const offset = reset ? 0 : state.viewer.offset;
  if (reset) { state.viewer.episodes = []; state.viewer.offset = 0; resetEpisodeView('Choose an episode to view its cameras and action samples.'); }
  $('viewer-load-more').disabled = true; $('viewer-load-more').textContent = 'Loading…';
  try {
    const result = await api.episodes(datasetId, { offset, limit: 300, task, signal: controller.signal });
    if (requestToken !== episodesRequestToken || state.viewer.datasetId !== datasetId || state.viewer.task !== task) return;
    const keepValue = $('viewer-episode').value;
    state.viewer.episodes = reset ? result.episodes || [] : state.viewer.episodes.concat(result.episodes || []);
    state.viewer.offset = Number(result.next_offset ?? (offset + (result.episodes || []).length));
    state.viewer.total = Number(result.total ?? state.viewer.total);
    updateEpisodeSelect(keepValue);
    $('viewer-load-more').hidden = state.viewer.offset >= state.viewer.total;
    $('viewer-load-more').disabled = false; $('viewer-load-more').textContent = 'Load more episodes';
    if (reset && state.viewer.episodes.length) {
      $('viewer-episode').value = String(state.viewer.episodes[0].episode_index);
      await loadEpisode(state.viewer.episodes[0].episode_index);
    } else if (reset) resetEpisodeView('No episodes match this task filter.');
  } catch (error) {
    if (error.name === 'AbortError') return;
    if (requestToken !== episodesRequestToken || state.viewer.datasetId !== datasetId || state.viewer.task !== task) return;
    $('viewer-load-more').hidden = true; $('viewer-load-more').disabled = false; $('viewer-load-more').textContent = 'Load more episodes';
    resetEpisodeView(`Could not load episode metadata: ${error.message}`);
  } finally {
    if (episodesRequestController === controller) episodesRequestController = null;
  }
}

async function openDataset(datasetId) {
  if (!datasetId) return;
  const token = ++state.viewer.requestToken;
  episodeToken += 1; pauseAndClearMedia();
  state.viewer.datasetId = datasetId; state.viewer.metadata = null; state.viewer.task = ''; state.viewer.episodes = [];
  $('dataset-overview').hidden = true; $('dataset-viewer').hidden = false;
  $('viewer-title').textContent = datasetId; $('viewer-subtitle').textContent = 'Loading dataset capabilities…';
  $('viewer-meta').replaceChildren(metaCard('Metadata', 'Loading…'));
  $('viewer-episode').replaceChildren(new Option('Loading episodes…', '')); $('viewer-episode').disabled = true;
  $('viewer-load-more').hidden = true; resetEpisodeView('Loading dataset metadata and episode list…');
  try {
    const metadata = await api.viewer(datasetId);
    if (token !== state.viewer.requestToken) return;
    state.viewer.metadata = metadata; renderViewerMetadata(metadata);
    await loadEpisodePage(true);
  } catch (error) {
    if (token !== state.viewer.requestToken) return;
    $('viewer-subtitle').textContent = 'Dataset metadata could not be loaded.';
    $('viewer-meta').replaceChildren(metaCard('Viewer error', 'Unavailable', error.message));
    resetEpisodeView(`Could not load dataset viewer: ${error.message}`);
  }
}

function closeViewer() {
  window.clearTimeout(episodeChangeTimer);
  episodesRequestController?.abort(); episodesRequestController = null;
  actionRequestController?.abort(); actionRequestController = null;
  episodeToken += 1; episodesRequestToken += 1; state.viewer.requestToken += 1; pauseAndClearMedia();
  state.viewer.datasetId = ''; state.viewer.metadata = null; state.viewer.episodes = [];
  $('dataset-viewer').hidden = true; $('dataset-overview').hidden = false;
}

function playEpisode() {
  if (!activeVideos.length) return;
  const record = masterRecord();
  const relative = record ? relTime(record.video) : 0;
  if (relative >= currentVideoDuration - .05) seekAll(0);
  activeVideos.forEach(item => {
    if (!item.video.paused) return;
    const target = recordStart(item) + Math.max(0, relative >= currentVideoDuration - .05 ? 0 : relative);
    item.video.currentTime = Math.min(target, Math.max(recordStart(item), recordPlayableEnd(item) - .02));
    item.video.play().catch(error => notify(`Could not play ${item.camera_key}: ${error.message}`, 'error'));
  });
}

function pauseEpisode() {
  syncingMedia = true; activeVideos.forEach(item => item.video.pause()); syncingMedia = false;
}

function stopEpisode() {
  pauseEpisode(); seekAll(0);
}

export function initDatasetViewer() {
  window.addEventListener('dataset:open', event => openDataset(event.detail));
  window.addEventListener('dataset:back', closeViewer);
  $('back-to-datasets').addEventListener('click', closeViewer);
  $('viewer-task').addEventListener('change', event => {
    state.viewer.task = event.target.value; loadEpisodePage(true);
  });
  $('viewer-episode').addEventListener('change', event => {
    window.clearTimeout(episodeChangeTimer);
    const selectedEpisode = event.target.value;
    episodeChangeTimer = window.setTimeout(() => loadEpisode(selectedEpisode), 60);
  });
  $('viewer-load-more').addEventListener('click', () => loadEpisodePage(false));
  $('viewer-play').addEventListener('click', playEpisode);
  $('viewer-pause').addEventListener('click', pauseEpisode);
  $('viewer-stop').addEventListener('click', stopEpisode);
  $('viewer-seek').addEventListener('input', event => seekAll(Number(event.target.value)));
  $('action-page-prev').addEventListener('click', () => renderActionCharts(state.viewer.actionPage - 1));
  $('action-page-next').addEventListener('click', () => renderActionCharts(state.viewer.actionPage + 1));
}
