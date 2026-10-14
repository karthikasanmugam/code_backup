function safeArray(value) {
  return Array.isArray(value) ? value : [];
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export function renderInspection(inspection, elements) {
  elements.section.hidden = false;
  if (elements.next) elements.next.hidden = false;
  const datasetRoot = String(inspection.dataset_root ?? "").replace(/[\\/]+$/, "");
  const datasetName = datasetRoot.split(/[\\/]/).filter(Boolean).pop() || datasetRoot;
  elements.path.textContent = datasetName ? `Dataset name: ${datasetName}` : "";
  elements.path.hidden = !datasetName;

  const metrics = [
    [inspection.total_episodes, "Episodes"],
    [inspection.total_frames, "Frames"],
    [inspection.fps, "FPS"],
    [inspection.state_dim, "State dims"],
    [inspection.action_dim, "Action dims"],
    [inspection.total_tasks, "Tasks"],
  ];

  elements.summary.innerHTML = metrics
    .map(
      ([value, label]) => `
    <div class="metric">
      <span class="metric-value">${escapeHtml(value)}</span>
      <span class="metric-label">${escapeHtml(label)}</span>
    </div>
  `,
    )
    .join("");

  const cameras = safeArray(inspection.cameras);

  elements.cameras.innerHTML = cameras.length
    ? cameras
        .map(
          (camera) => `
      <div class="list-item">
        <span class="name">${escapeHtml(camera.key)}</span>
        <span class="meta">
          ${escapeHtml(camera.width ?? "?")} ×
          ${escapeHtml(camera.height ?? "?")}
        </span>
      </div>
    `,
        )
        .join("")
    : `<div class="empty">No cameras declared.</div>`;

  const actions = safeArray(inspection.action_names);

  elements.actions.innerHTML = actions.length
    ? actions
        .map(
          (name, index) => `
      <div class="list-item">
        <span class="name">${escapeHtml(name)}</span>
        <span class="meta">${index}</span>
      </div>
    `,
        )
        .join("")
    : `<div class="empty">No action names available.</div>`;

  const states = safeArray(inspection.state_names);

  elements.states.innerHTML = states.length
    ? states
        .map(
          (name, index) => `
      <div class="list-item">
        <span class="name">${escapeHtml(name)}</span>
        <span class="meta">${index}</span>
      </div>
    `,
        )
        .join("")
    : `<div class="empty">No state names available.</div>`;

  const tasks = safeArray(inspection.tasks);

  elements.tasks.innerHTML = tasks.length
    ? tasks
        .map(
          (task) => `
      <div class="list-item">
        <span class="name">${escapeHtml(task)}</span>
      </div>
    `,
        )
        .join("")
    : `<div class="empty">No task text available.</div>`;
}