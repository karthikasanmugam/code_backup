function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function getState(run) {
  return String(
    run?.state ??
      run?.status ??
      "unknown",
  ).toLowerCase();
}

export function isTerminalState(state) {
  return [
    "completed",
    "cancelled",
    "canceled",
    "failed",
    "error",
  ].includes(String(state).toLowerCase());
}

// Active = engine is actually processing; drives the energetic run animation.
// Idle/waiting/terminal states fall back to the calm "waiting" sonar pulse.
export function isActiveState(state) {
  return [
    "running",
    "processing",
    "in_progress",
    "validating",
    "starting",
    "active",
  ].includes(String(state).toLowerCase());
}

function extractProgress(run) {
  const progress =
    run?.progress ??
    run?.metadata ??
    {};

  const completed =
    Number(
      progress.completed_episodes ??
        progress.episodes_completed ??
        progress.completed ??
        0,
    );

  const total =
    Number(
      progress.total_episodes ??
        progress.episodes_total ??
        progress.total ??
        0,
    );

  let percent =
    Number(
      progress.percent ??
        progress.percentage ??
        run?.percent ??
        0,
    );

  if (
    total > 0 &&
    percent <= 0
  ) {
    percent = (completed / total) * 100;
  }

  return {
    completed,
    total,
    percent: Math.max(
      0,
      Math.min(100, percent),
    ),
    phase:
      progress.phase ??
      progress.current_phase ??
      "",
    check:
      progress.check ??
      progress.current_check ??
      "",
    episode:
      progress.episode ??
      progress.current_episode ??
      null,
  };
}

export function renderRunState(
  run,
  elements,
) {
  const state = getState(run);
  const progress = extractProgress(run);

  elements.state.textContent = state;
  elements.state.className =
    `badge run-state-${state}`;

  if (run?.run_id) {
    elements.runId.textContent =
      run.run_id;
  }

  elements.progress.value =
    progress.percent;

  if (progress.total > 0) {
    elements.progressText.textContent =
      `${progress.completed} / ${progress.total} episodes`;
  } else {
    elements.progressText.textContent =
      `${Math.round(progress.percent)}%`;
  }

  const phaseText = progress.phase
    ? `Phase: ${progress.phase}`
    : "";

  const checkText = progress.check
    ? `Check: ${progress.check}`
    : "";

  const episodeText =
    progress.episode !== null &&
    progress.episode !== undefined
      ? `Episode: ${progress.episode}`
      : "";

  const stateText = isTerminalState(state)
    ? state === "completed"
      ? "Validation completed."
      : `Validation ended with state: ${state}.`
    : "Validation running…";

  elements.progressLabel.textContent =
    [
      phaseText,
      checkText,
      episodeText,
    ]
      .filter(Boolean)
      .join(" · ") ||
    stateText;

  elements.stopButton.disabled =
    isTerminalState(state);

  const artEl = document.querySelector(".run-header-art");
  if (artEl) {
    artEl.classList.toggle(
      "is-running",
      isActiveState(state),
    );
  }
}

export function renderRunSummary(
  summary,
  container,
) {
  if (!summary) {
    container.innerHTML = "";
    return;
  }

  const metrics = [
    [
      summary.total_episodes,
      "Episodes",
    ],
    [
      summary.pass_count,
      "Pass",
    ],
    [
      summary.review_count,
      "Review",
    ],
    [
      summary.reject_count,
      "Reject",
    ],
    [
      summary.hard_failures,
      "HARD",
    ],
    [
      summary.soft_warnings,
      "SOFT",
    ],
  ];

  container.innerHTML = metrics
    .map(
      ([value, label]) => `
        <div class="metric">
          <span class="metric-value">
            ${escapeHtml(
              value ?? 0,
            )}
          </span>

          <span class="metric-label">
            ${escapeHtml(label)}
          </span>
        </div>
      `,
    )
    .join("");
}

export function renderRunResults(results, container) {
  const episodes = Array.isArray(results?.episodes)
    ? results.episodes
    : Array.isArray(results?.episode_results) ? results.episode_results : [];
  if (!episodes.length) {
    container.innerHTML = `<div class="empty">No episode results returned.</div>`;
    return;
  }
  const cards = episodes.map((episode) => {
    const checks = Array.isArray(episode.results) ? episode.results : [];
    const status = String(episode.classification ?? episode.status ?? "unknown").toLowerCase();
    const id = episode.episode_index ?? episode.episode;
    const hard = Number(episode.hard_count ?? checks.filter((result) => String(result.severity).toUpperCase() === "HARD").length);
    const soft = Number(episode.soft_count ?? checks.filter((result) => String(result.severity).toUpperCase() === "SOFT").length);
    const findings = checks.length || hard + soft;
    return `<article class="result-item" data-episode="${escapeHtml(id)}
    
    ">
      <div class="result-heading">
        <label class="episode-select"><input type="checkbox" class="episode-checkbox" data-episode=" ${escapeHtml(id)}"> 
        </label>
        <strong>Episode ${escapeHtml(id)}</strong>
        <span class="badge run-state-${escapeHtml(status)}">${ escapeHtml( episode.classification ??  episode.status ?? " unknown")}</span>
      </div>
      <div class="result-meta">Score: ${escapeHtml(episode.score ?? "n/a")} · Findings: ${findings} · HARD: ${hard} · SOFT: ${soft}</div>
      ${checks.length ? `<ul class="result-findings">${checks.map((result) => `<li><strong>${escapeHtml(result.severity ?? "INFO")}</strong> ${escapeHtml(result.check ?? "unknown")}: ${escapeHtml(result.message ?? "")}</li>`).join("")}</ul>` : findings ? `<button class="secondary-button result-detail-button" type="button" data-episode="${escapeHtml(id)}">Load findings</button><div class="result-detail" hidden></div>` : `<div class="empty">No findings.</div>`}
    </article>`;
  }).join("");
  container.innerHTML = cards;
}

export async function waitForCompletion({
  apiClient,
  runId,
  elements,
  intervalMs = 1000,
  shouldStop = () => false,
}) {
  while (true) {
    const run =
      await apiClient.getRun(runId);

    renderRunState(
      run,
      elements,
    );

    const state =
      getState(run);

    if (isTerminalState(state)) {
      return run;
    }

    await new Promise((resolve) =>
      setTimeout(
        resolve,
        intervalMs,
      ),
    );
  }
}

export function showValidationSection(
  elements,
) {
  elements.section.hidden = false;
}

export function resetRunSection(elements) {
  const artEl = document.querySelector(".run-header-art");
  if (artEl) artEl.classList.add("is-running");

  elements.state.textContent =
    "starting";

  elements.state.className =
    "badge badge-neutral";

  elements.runId.textContent = "";

  elements.progress.value = 0;

  elements.progressLabel.textContent =
    "Starting validation…";

  elements.progressText.textContent =
    "Working…";

  elements.stopButton.disabled =
    false;

  elements.summary.innerHTML = "";
}