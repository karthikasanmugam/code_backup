export class ApiError extends Error {
  constructor(status, detail) {
    super(`API request failed (${status})`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export class ValidatorApiClient {
  constructor(baseUrl = "") {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  async request(path, options = {}) {
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...options,
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
    });

    const contentType =
      response.headers.get("content-type") || "";

    const payload = contentType.includes("application/json")
      ? await response.json()
      : await response.text();

    if (!response.ok) {
      const detail =
        payload &&
        typeof payload === "object" &&
        "detail" in payload
          ? payload.detail
          : payload;

      throw new ApiError(response.status, detail);
    }

    return payload;
  }

  health() {
    return this.request("/api/health");
  }

  inspectDataset(datasetRoot) {
    return this.request("/api/datasets/inspect", {
      method: "POST",
      body: JSON.stringify({
        dataset_root: datasetRoot,
      }),
    });
  }

  deleteEpisodes(datasetRoot, episodes) {
    return this.request("/api/datasets/delete-episodes", {
      method: "POST",
      body: JSON.stringify({ dataset_root: datasetRoot, episodes }),
    });
  }

  getChecks(datasetRoot = undefined) {
    const query = datasetRoot
      ? `?dataset_root=${encodeURIComponent(datasetRoot)}`
      : "";

    return this.request(`/api/checks${query}`);
  }

  resolveConfig({
    datasetRoot,
    profile = "default",
    overrides = {},
    enabledChecks = {},
  }) {
    return this.request("/api/config/resolve", {
      method: "POST",
      body: JSON.stringify({
        dataset_root: datasetRoot,
        profile,
        overrides,
        enabled_checks: enabledChecks,
      }),
    });
  }

  createRun({
    datasetRoot,
    profile = "default",
    overrides = {},
    enabledChecks = {},
  }) {
    return this.request("/api/runs", {
      method: "POST",
      body: JSON.stringify({
        dataset_root: datasetRoot,
        config: {
          dataset_root: datasetRoot,
          profile,
          overrides,
          enabled_checks: enabledChecks,
        },
      }),
    });
  }

  getRun(runId) {
    return this.request(
      `/api/runs/${encodeURIComponent(runId)}`,
    );
  }

  stopRun(runId) {
    return this.request(
      `/api/runs/${encodeURIComponent(runId)}/stop`,
      {
        method: "POST",
      },
    );
  }

  getSummary(runId) {
    return this.request(
      `/api/runs/${encodeURIComponent(runId)}/summary`,
    );
  }

  getResults(runId) {
    return this.request(
      `/api/runs/${encodeURIComponent(runId)}/results`,
    );
  }

  getEpisode(runId, episodeId) {
    return this.request(
      `/api/runs/${encodeURIComponent(runId)}/episodes/${encodeURIComponent(episodeId)}`,
    );
  }

  getEvidence(runId, episodeId) {
    return this.request(
      `/api/runs/${encodeURIComponent(runId)}/episodes/${encodeURIComponent(episodeId)}/evidence`,
    );
  }
}

export const apiClient = new ValidatorApiClient(
  globalThis.VALIDATOR_API_BASE || "",
);