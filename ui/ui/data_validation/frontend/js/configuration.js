function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function controlPath(control) {
  return Array.isArray(control.config_path)
    ? control.config_path
    : [];
}

function inputValue(control) {
  if (control.default === null || control.default === undefined) {
    return "";
  }

  if (Array.isArray(control.default)) {
    return control.default;
  }

  return control.default;
}

function renderControl(control, definition, options) {
  const id = `control-${definition.id}-${control.id}`
    .replace(/[^a-zA-Z0-9_-]/g, "-");

  const disabled =
    control.binding_status === "pending_engine_wiring";

  const path = JSON.stringify(controlPath(control));
  const value = inputValue(control);

  const min =
    control.minimum !== null &&
    control.minimum !== undefined
      ? `min="${escapeHtml(control.minimum)}"`
      : "";

  const max =
    control.maximum !== null &&
    control.maximum !== undefined
      ? `max="${escapeHtml(control.maximum)}"`
      : "";

  const step =
    control.step !== null &&
    control.step !== undefined
      ? `step="${escapeHtml(control.step)}"`
      : "";

  const disabledAttr = disabled ? "disabled" : "";

  const type = control.control_type;

  if (
    type === "camera_select" ||
    type === "feature_select"
  ) {
    const selectOptions = options.length
      ? options
          .map((option) => {
            const selected =
              String(option) === String(value)
                ? "selected"
                : "";

            return `
              <option
                value="${escapeHtml(option)}"
                ${selected}
              >
                ${escapeHtml(option)}
              </option>
            `;
          })
          .join("")
      : `
          <option value="">
            No options detected
          </option>
        `;

    return `
      <div class="config-control">
        <label for="${id}">
          ${escapeHtml(control.label || control.id)}
        </label>

        <select
          id="${id}"
          data-config-path='${path}'
          ${disabledAttr}
        >
          ${selectOptions}
        </select>

        ${
          control.unit
            ? `<small>${escapeHtml(control.unit)}</small>`
            : ""
        }

        ${
          control.description
            ? `<div class="control-help">${escapeHtml(
                control.description,
              )}</div>`
            : ""
        }
      </div>
    `;
  }

  if (type === "feature_multi_select") {
    const selectedValues = Array.isArray(value)
      ? value.map(String)
      : [];

    const multiOptions = options.length
      ? options
          .map((option) => {
            const checked = selectedValues.includes(
              String(option),
            )
              ? "checked"
              : "";

            return `
              <label class="checkbox-option">
                <input
                  type="checkbox"
                  value="${escapeHtml(option)}"
                  data-multi-path='${path}'
                  ${checked}
                  ${disabledAttr}
                >
                <span>${escapeHtml(option)}</span>
              </label>
            `;
          })
          .join("")
      : `<div class="empty">No feature names detected.</div>`;

    return `
      <div class="config-control">
        <label>
          ${escapeHtml(control.label || control.id)}
        </label>

        <div class="checkbox-list">
          ${multiOptions}
        </div>

        ${
          control.description
            ? `<div class="control-help">${escapeHtml(
                control.description,
              )}</div>`
            : ""
        }
      </div>
    `;
  }

  const htmlType =
    type === "integer" || type === "number"
      ? "number"
      : "text";

  return `
    <div class="config-control">
      <label for="${id}">
        ${escapeHtml(control.label || control.id)}
      </label>

      <input
        id="${id}"
        type="${htmlType}"
        value="${escapeHtml(value)}"
        data-config-path='${path}'
        ${min}
        ${max}
        ${step}
        ${disabledAttr}
      >

      ${
        control.unit
          ? `<small>${escapeHtml(control.unit)}</small>`
          : ""
      }

      ${
        control.description
          ? `<div class="control-help">${escapeHtml(
              control.description,
            )}</div>`
          : ""
      }

      ${
        disabled
          ? `<div class="pending-note">
               UI control waiting for engine wiring
             </div>`
          : ""
      }
    </div>
  `;
}

function optionsForControl(control, inspection) {
  if (!inspection) {
    return Array.isArray(control.options)
      ? control.options
      : [];
  }

  if (control.control_type === "camera_select") {
    return [
      "auto",
      ...(Array.isArray(inspection.cameras)
        ? inspection.cameras.map((camera) => camera.key)
        : []),
    ];
  }

  if (
    control.control_type === "feature_select" ||
    control.control_type === "feature_multi_select"
  ) {
    return Array.isArray(inspection.action_names)
      ? inspection.action_names
      : [];
  }

  return Array.isArray(control.options)
    ? control.options
    : [];
}

export function renderCheckDefinitions(
  definitions,
  container,
  inspection = null,
) {
  const checks = Array.isArray(definitions)
    ? definitions
    : [];

  if (!checks.length) {
    container.innerHTML =
      `<div class="empty">
        No validation controls returned by the backend.
      </div>`;
    return;
  }

  container.innerHTML = checks
    .map((definition) => {
      const controls = Array.isArray(definition.controls)
        ? definition.controls
        : [];

      return `
        <article
          class="check-card${controls.length ? "" : " check-card-no-controls"}"
          data-check-id="${escapeHtml(definition.id)}"
        >
          ${controls.length
            ? `<div class="config-controls">
                ${controls
                  .map((control) =>
                    renderControl(
                      control,
                      definition,
                      optionsForControl(control, inspection),
                    ),
                  )
                  .join("")}
              </div>`
            : ""}
        </article>
      `;
    })
    .join("");
}

function setNestedValue(target, path, value) {
  if (!path.length) {
    return;
  }

  let current = target;

  for (let index = 0; index < path.length - 1; index += 1) {
    const key = path[index];

    if (
      !current[key] ||
      typeof current[key] !== "object" ||
      Array.isArray(current[key])
    ) {
      current[key] = {};
    }

    current = current[key];
  }

  current[path[path.length - 1]] = value;
}

function parseInputValue(input) {
  if (input.type === "number") {
    if (input.value === "") {
      return null;
    }

    const number = Number(input.value);
    return Number.isNaN(number)
      ? null
      : number;
  }

  return input.value;
}

export function collectConfiguration(
  container,
) {
  const overrides = {};
  const enabledChecks = {};

  container
    .querySelectorAll(".check-card[data-check-id]")
    .forEach((card) => {
      enabledChecks[card.dataset.checkId] = true;
    });

  container
    .querySelectorAll(
      "input[data-config-path], select[data-config-path]",
    )
    .forEach((input) => {
      if (input.disabled) {
        return;
      }

      let path;

      try {
        path = JSON.parse(input.dataset.configPath);
      } catch {
        return;
      }

      const value = parseInputValue(input);
      setNestedValue(overrides, path, value);
    });

  const groupedMultiSelects = new Map();

  container
    .querySelectorAll(
      "input[type='checkbox'][data-multi-path]",
    )
    .forEach((input) => {
      if (input.disabled) {
        return;
      }

      const path = input.dataset.multiPath;

      if (!groupedMultiSelects.has(path)) {
        groupedMultiSelects.set(path, []);
      }

      if (input.checked) {
        groupedMultiSelects
          .get(path)
          .push(input.value);
      }
    });

  for (const [serializedPath, values] of groupedMultiSelects) {
    let path;

    try {
      path = JSON.parse(serializedPath);
    } catch {
      continue;
    }

    setNestedValue(
      overrides,
      path,
      values,
    );
  }

  return {
    overrides,
    enabledChecks,
  };
}