import {
  ApiError,
  apiClient,
} from "./api.js?v=2";

import {
  renderCheckDefinitions,
  collectConfiguration,
} from "./configuration.js?v=2";

import {
  renderInspection,
} from "./dataset.js?v=2";

import {
  renderRunState,
  renderRunSummary,
  renderRunResults,
  waitForCompletion,
  resetRunSection,
} from "./validation.js?v=2";

const elements = {
  form:
    document.querySelector(
      "#inspect-form",
    ),

  input:
    document.querySelector(
      "#dataset-root",
    ),

  inspectButton:
    document.querySelector(
      "#inspect-button",
    ),

  error:
    document.querySelector(
      "#form-error",
    ),

  health:
    document.querySelector(
      "#health-badge",
    ),

  inspection:
    document.querySelector(
      "#inspection-section",
    ),

  path:
    document.querySelector(
      "#dataset-path",
    ),

  summary:
    document.querySelector(
      "#summary-grid",
    ),

  cameras:
    document.querySelector(
      "#camera-list",
    ),

  actions:
    document.querySelector(
      "#action-list",
    ),

  states:
    document.querySelector(
      "#state-list",
    ),

  tasks:
    document.querySelector(
      "#task-list",
    ),

  next:
    document.querySelector(
      "#next-step",
    ),

  loadChecks:
    document.querySelector(
      "#load-checks-button",
    ),

  checks:
    document.querySelector(
      "#checks-section",
    ),

  checkList:
    document.querySelector(
      "#checks-list",
    ),

  config:
    document.querySelector(
      "#config-section",
    ),

  configHost:
    document.querySelector(
      "#validation-form-host",
    ),

  run:
    document.querySelector(
      "#run-section",
    ),

  runState:
    document.querySelector(
      "#run-state",
    ),

  runId:
    document.querySelector(
      "#run-id",
    ),

  progressLabel:
    document.querySelector(
      "#run-progress-label",
    ),

  progress:
    document.querySelector(
      "#run-progress",
    ),

  progressText:
    document.querySelector(
      "#run-progress-text",
    ),

  stopRun:
    document.querySelector(
      "#stop-run-button",
    ),

  runSummary:
    document.querySelector(
      "#run-summary",
    ),

  runResults:
    document.querySelector(
      "#run-results",
    ),

  deleteStatus:
    document.querySelector(
      "#delete-status",
    ),

  checkSearch:
    document.querySelector(
      ".search-input",
    ),
};

const inspectionElements = {
  section:
    elements.inspection,

  next:
    elements.next,

  path:
    elements.path,

  summary:
    elements.summary,

  cameras:
    elements.cameras,

  actions:
    elements.actions,

  states:
    elements.states,

  tasks:
    elements.tasks,
};

const runElements = {
  section:
    elements.run,

  state:
    elements.runState,

  runId:
    elements.runId,

  progressLabel:
    elements.progressLabel,

  progress:
    elements.progress,

  progressText:
    elements.progressText,

  stopButton:
    elements.stopRun,

  summary:
    elements.runSummary,
};

let currentDatasetRoot = "";
let currentInspection = null;
let currentRunId = null;
let validationStopRequested = false;
let selectedEpisodes = new Set();

function setupEpisodeSelection({ reset = false } = {}) {
  if (reset) selectedEpisodes.clear();
  const count = document.querySelector("#selected-episodes");
  const removeButton = document.querySelector("#delete-selected-episodes");
  const refresh = () => {
    if (count) count.textContent = `${selectedEpisodes.size} selected`;
    if (removeButton) removeButton.disabled = selectedEpisodes.size === 0;
  };
  if (!elements.runResults.dataset.selectionBound) {
    elements.runResults.addEventListener("change", (event) => {
      const checkbox = event.target.closest(".episode-checkbox");
      if (!checkbox) return;
      const episode = Number(checkbox.dataset.episode);
      if (checkbox.checked) selectedEpisodes.add(episode);
      else selectedEpisodes.delete(episode);
      refresh();
    });
    elements.runResults.dataset.selectionBound = "true";
  }
  if (removeButton && !removeButton.dataset.deleteBound) {
    removeButton.dataset.deleteBound = "true";
    removeButton.onclick = async () => {
    if (!selectedEpisodes.size || !currentDatasetRoot) return;
    const episodes = [...selectedEpisodes].sort((a, b) => b - a);
    if (!window.confirm(`Delete ${episodes.length} selected episode(s)? This cannot be undone.`)) return;
    removeButton.disabled = true;
    removeButton.textContent = "Deleting…";
    const checkboxes = [...elements.runResults.querySelectorAll(".episode-checkbox")];
    checkboxes.forEach((checkbox) => { checkbox.disabled = true; });
    setDeleteStatus(`Deleting ${episodes.length} episode(s). Rebuilding shared parquet and video files; this may take a while. Keep this page open.`, "working");
    let deletionResult = null;
    try {
      deletionResult = await apiClient.deleteEpisodes(currentDatasetRoot, episodes);
      const inspection = await apiClient.inspectDataset(currentDatasetRoot);
      currentInspection = inspection;
      renderInspection(inspection, inspectionElements);
      selectedEpisodes.clear();
      const selectedCount = document.querySelector("#selected-episodes");
      if (selectedCount) selectedCount.textContent = "0 selected";
      currentRunId = null;
      elements.runId.textContent = "";
      elements.runState.textContent = "dataset updated";
      elements.runState.className = "badge badge-ok";
      elements.progress.value = 0;
      elements.progressLabel.textContent = "Dataset changed — run validation again.";
      elements.progressText.textContent = "Previous results cleared after deletion.";
      elements.stopRun.disabled = true;
      elements.runSummary.innerHTML = "";
      elements.runResults.innerHTML = `<div class="empty">Dataset changed. Start a new validation run to see current results.</div>`;
      setupEpisodeSelection({ reset: true });
      setDeleteStatus(`Deleted episode${episodes.length === 1 ? "" : "s"} ${episodes.join(", ")}. Dataset refreshed: ${inspection.total_episodes ?? deletionResult.remaining_episodes ?? "updated"} episode(s) remain. Old validation results were cleared.`, "success");
      setActiveStep(2);
    } catch (error) {
      if (deletionResult) {
        currentRunId = null;
        elements.runSummary.innerHTML = "";
        elements.runResults.innerHTML = `<div class="empty">Dataset changed. Inspect the dataset again and start a new validation run.</div>`;
        setDeleteStatus(`Deletion succeeded, but refreshing dataset details failed: ${formatApiError(error)}. Do not retry this deletion; inspect the dataset again.`, "error");
      } else {
        setDeleteStatus(`Deletion failed: ${formatApiError(error)} Selected episodes were not cleared from the UI; you can retry.`, "error");
        removeButton.disabled = false;
        elements.runResults.querySelectorAll(".episode-checkbox").forEach((checkbox) => { checkbox.disabled = false; });
      }
    } finally { removeButton.textContent = "Delete selected"; }
    };
  }
  refresh();
}

async function loadAllRunResults() {
  if (!currentRunId) return;
  try {
    const results = await apiClient.getResults(currentRunId);
    renderRunResults(results, elements.runResults);
    setupEpisodeSelection();
  } catch (error) {
    showError(formatApiError(error));
  }
}

function setDeleteStatus(message, state) {
  if (!elements.deleteStatus) return;
  elements.deleteStatus.textContent = message;
  elements.deleteStatus.dataset.state = state;
  elements.deleteStatus.hidden = false;
}

function setHealth(ok) {
  elements.health.textContent = ok
    ? "API: connected"
    : "API: unavailable";

  elements.health.className =
    `badge ${
      ok
        ? "badge-ok"
        : "badge-error"
    }`;
}

function showError(message) {
  elements.error.textContent =
    message;

  elements.error.hidden = false;
}

function clearError() {
  elements.error.textContent = "";
  elements.error.hidden = true;
}

function setActiveStep(step) {
  const pages = [
    "dataset-selection",
    "inspection-section",
    "config-section",
    "run-section",
  ];

  document.querySelectorAll(".step-link").forEach((link, index) => {
    link.classList.toggle("active", index + 1 === step);
    link.setAttribute("aria-current", index + 1 === step ? "step" : "false");
  });

  pages.forEach((id, index) => {
    const page = document.getElementById(id);
    if (!page) return;
    page.hidden = index + 1 !== step;
    page.classList.toggle("active-page", index + 1 === step);
  });

  // The checks registry is an intermediate data source. Its controls are copied
  // into the configuration page, so it should never create another visible page.
  if (elements.checks) elements.checks.hidden = true;
}

function installStepNavigation() {
  document.querySelectorAll(".step-link").forEach((link, index) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      setActiveStep(index + 1);
    });
  });
  setActiveStep(1);
}

function filterChecks(event) {
  const query = event.target.value.trim().toLowerCase();

  elements.configHost
    .querySelectorAll(".check-card")
    .forEach((card) => {
      card.hidden = query !== "" &&
        !card.textContent.toLowerCase().includes(query);
    });
}

function formatApiError(error) {
  if (error instanceof ApiError) {
    if (
      typeof error.detail ===
      "string"
    ) {
      return error.detail;
    }

    return `Request failed with HTTP ${error.status}.`;
  }

  return error instanceof Error
    ? error.message
    : "Unexpected error.";
}

async function checkHealth() {
  try {
    await apiClient.health();
    setHealth(true);
  } catch {
    setHealth(false);
  }
}

async function handleInspect(event) {
  event.preventDefault();

  clearError();

  const root =
    elements.input.value.trim();

  if (!root) {
    showError(
      "Enter a dataset root path.",
    );
    return;
  }

  elements.inspectButton.disabled =
    true;

  elements.inspectButton.textContent =
    "Inspecting…";

  try {
    const inspection =
      await apiClient.inspectDataset(
        root,
      );

    currentDatasetRoot = root;
    currentInspection =
      inspection;

    renderInspection(
      inspection,
      inspectionElements,
    );

    setActiveStep(2);
    // Preload validation controls for externally selected datasets without
    // navigating away from the Inspect Dataset page.
    await handleLoadChecks({ navigate: !window.__validatorAutoInspectStarted });
  } catch (error) {
    if (window.__validatorAutoInspectStarted) {
      setActiveStep(2);
    }
    showError(
      formatApiError(error),
    );
  } finally {
    elements.inspectButton.disabled =
      false;

    elements.inspectButton.textContent =
      "Inspect dataset";
  }
}

async function handleLoadChecks({ navigate = true } = {}) {
  if (!currentDatasetRoot) {
    showError(
      "Inspect a dataset first.",
    );
    return;
  }

  clearError();
  if (navigate) setActiveStep(3);

  if (elements.loadChecks) elements.loadChecks.disabled = true;
  if (elements.loadChecks) elements.loadChecks.textContent = "Loading…";

  try {
    const definitions =
      await apiClient.getChecks(
        currentDatasetRoot,
      );

    if (navigate) elements.config.hidden = false;
    renderCheckDefinitions(
      definitions,
      elements.configHost,
      currentInspection,
    );

    attachStartValidationButton();
    if (navigate) setActiveStep(3);
  } catch (error) {
    if (window.__validatorAutoInspectStarted) {
      setActiveStep(2);
    }
    showError(
      formatApiError(error),
    );
  } finally {
    if (elements.loadChecks) elements.loadChecks.disabled = false;
    if (elements.loadChecks) elements.loadChecks.textContent = "Load validation controls";
  }
}

function attachStartValidationButton() {
  let footer =
    elements.config.querySelector(
      ".configuration-footer",
    );

  if (footer) {
    return;
  }

  footer =
    document.createElement(
      "div",
    );

  footer.className =
    "configuration-footer";

  footer.innerHTML = `
    <button
      id="resolve-config-button"
      type="button"
      class="secondary-button"
    >
      Validate configuration
    </button>

    <button
      id="start-validation-button"
      type="button"
    >
      Start validation
    </button>

    <div
      id="configuration-message"
      class="form-error"
      hidden
    ></div>
  `;

  elements.config.appendChild(
    footer,
  );

  footer
    .querySelector(
      "#resolve-config-button",
    )
    .addEventListener(
      "click",
      handleResolveConfig,
    );

  footer
    .querySelector(
      "#start-validation-button",
    )
    .addEventListener(
      "click",
      handleStartValidation,
    );
}

function getConfigurationRequest() {
  const {
    overrides,
    enabledChecks,
  } =
    collectConfiguration(
      elements.configHost,
    );

  return {
    datasetRoot:
      currentDatasetRoot,

    profile: "default",

    overrides,

    enabledChecks,
  };
}

function showConfigMessage(
  message,
  isError = true,
) {
  const messageElement =
    document.querySelector(
      "#configuration-message",
    );

  if (!messageElement) {
    return;
  }

  messageElement.textContent =
    message;

  messageElement.hidden = false;

  messageElement.style.color =
    isError
      ? ""
      : "inherit";
}

async function resolveConfiguration() {
  const request =
    getConfigurationRequest();

  const response =
    await apiClient.resolveConfig(
      request,
    );

  if (
    response.valid === false
  ) {
    const message =
      Array.isArray(
        response.errors,
      )
        ? response.errors.join(
            " | ",
          )
        : "Configuration is invalid.";

    throw new Error(
      message,
    );
  }

  return response;
}

async function handleResolveConfig() {
  clearError();

  try {
    const response =
      await resolveConfiguration();

    const warningText =
      Array.isArray(
        response.warnings,
      ) &&
      response.warnings.length
        ? response.warnings.join(
            " | ",
          )
        : "Configuration is valid.";

    showConfigMessage(
      warningText,
      false,
    );
  } catch (error) {
    showConfigMessage(
      formatApiError(error),
      true,
    );
  }
}

async function handleStartValidation() {
  clearError();

  const startButton =
    document.querySelector(
      "#start-validation-button",
    );

  if (!startButton) {
    return;
  }

  startButton.dataset.runStarting = "true";
  startButton.disabled = true;
  startButton.textContent =
    "Preparing…";

  try {
    const config =
      await resolveConfiguration();

    if (
      config.valid === false
    ) {
      throw new Error(
        "Configuration validation failed.",
      );
    }

    startButton.textContent =
      "Starting…";

    const run =
      await apiClient.createRun(
        getConfigurationRequest(),
      );

    currentRunId = run.run_id;
    validationStopRequested = false;
    selectedEpisodes.clear();
    elements.runResults.innerHTML = "";

    elements.run.hidden =
      false;

    setActiveStep(4);

    resetRunSection(
      runElements,
    );

    renderRunState(
      run,
      runElements,
    );

    const finalRun =
      await waitForCompletion({
        apiClient,
        runId: currentRunId,
        elements:
          runElements,
        shouldStop: () => validationStopRequested,
      });

    if (
      validationStopRequested &&
      ["cancelled", "canceled"].includes(
        String(finalRun?.state ?? "").toLowerCase()
      )
    ) {
      runElements.progressLabel.textContent = "Validation stopped.";
      runElements.progressText.textContent = "Run stopped. Press Start validation to begin a fresh run.";
      return;
    }

    const summary =
      await apiClient.getSummary(
        currentRunId,
      );

    renderRunSummary(
      summary,
      elements.runSummary,
    );

    setupEpisodeSelection({ reset: true });
    await loadAllRunResults();

    if (
      finalRun?.state ===
      "completed"
    ) {
      showConfigMessage(
        "Validation completed.",
        false,
      );
    } else {
      showConfigMessage(
        `Validation ended with state: ${finalRun?.state ?? "unknown"}.`,
        true,
      );
    }
  } catch (error) {
    showError(
      formatApiError(error),
    );

  } finally {
    delete startButton.dataset.runStarting;
    startButton.disabled = false;
    startButton.textContent = "Start validation";
  }
}

async function handleStopValidation() {
  if (!currentRunId) {
    return;
  }

  validationStopRequested = true;
  runElements.progressLabel.textContent = "Stopping validation…";
  elements.stopRun.disabled =
    true;

  try {
    const run = await apiClient.stopRun(currentRunId);
    validationStopRequested = true;
    renderRunState(run, runElements);
    runElements.progressLabel.textContent = "Waiting for validation to stop…";
    runElements.stopButton.disabled = true;
  } catch (error) {
    showError(
      formatApiError(error),
    );

    validationStopRequested = false;
    elements.stopRun.disabled =
      false;
  }
}

if (elements.form) {
  elements.form.addEventListener("submit", handleInspect);
  if (globalThis.VALIDATOR_DATASET) {
    elements.input.value = globalThis.VALIDATOR_DATASET;
    elements.form.requestSubmit();
  }
}

if (elements.loadChecks) {
  elements.loadChecks.addEventListener("click", handleLoadChecks);
}

if (elements.runResults) {
  elements.runResults.addEventListener("click", async (event) => {
    const button = event.target.closest(".result-detail-button");
    if (!button || !currentRunId) return;
    const episodeId = Number(button.dataset.episode);
    const panel = button.parentElement.querySelector(".result-detail");
    if (!panel || button.dataset.loading === "true") return;
    button.dataset.loading = "true";
    button.disabled = true;
    button.textContent = "Loading findings…";
    try {
      const episode = await apiClient.getEpisode(currentRunId, episodeId);
      const findings = Array.isArray(episode.results) ? episode.results : [];
      panel.replaceChildren();
      if (!findings.length) {
        panel.textContent = "No detailed findings.";
      } else {
        const list = document.createElement("ul");
        list.className = "result-findings";
        for (const finding of findings) {
          const item = document.createElement("li");
          const severity = document.createElement("strong");
          severity.textContent = finding.severity ?? "INFO";
          item.append(severity, document.createTextNode(` ${finding.check ?? "unknown"}: ${finding.message ?? ""}`));
          list.append(item);
        }
        panel.append(list);
      }
      panel.hidden = false;
      button.hidden = true;
    } catch (error) {
      showError(formatApiError(error));
      button.disabled = false;
      button.dataset.loading = "false";
      button.textContent = "Retry loading findings";
    }
  });
}

if (elements.stopRun) {
  elements.stopRun.addEventListener(
    "click",
    handleStopValidation,
  );
}

if (elements.checkSearch) {
  elements.checkSearch.addEventListener(
    "input",
    filterChecks,
  );
}

installStepNavigation();

checkHealth();