(function initAdminPingTest() {
  const dataEl = document.getElementById('ping-models-data');
  const providerSelect = document.getElementById('ping-provider');
  const modelSelect = document.getElementById('ping-model');
  const keySelect = document.getElementById('ping-key-index');
  const runBtn = document.getElementById('ping-run-btn');
  const resultEl = document.getElementById('ping-result');

  if (!dataEl || !providerSelect || !modelSelect || !keySelect || !runBtn || !resultEl) {
    return;
  }

  let catalog = {};
  try {
    catalog = JSON.parse(dataEl.textContent || '{}');
  } catch {
    catalog = {};
  }

  function setResult(message, ok) {
    resultEl.textContent = message;
    resultEl.classList.toggle('ping-result-ok', ok === true);
    resultEl.classList.toggle('ping-result-error', ok === false);
    resultEl.classList.toggle('ping-result-pending', ok == null);
  }

  function populateModels() {
    const provider = providerSelect.value;
    const models = catalog[provider] || [];
    const defaults = catalog.defaults || {};
    const preferred =
      provider === 'gemini' ? defaults.gemini_text : defaults.groq_text;

    modelSelect.innerHTML = '';
    models.forEach((model) => {
      const option = document.createElement('option');
      option.value = model.id;
      option.textContent = `${model.label} (${model.id})`;
      modelSelect.appendChild(option);
    });

    if (preferred && models.some((m) => m.id === preferred)) {
      modelSelect.value = preferred;
    }
  }

  function populateKeys() {
    const provider = providerSelect.value;
    const keys = catalog[`${provider}_keys`] || [];
    keySelect.innerHTML = '';
    if (!keys.length) {
      const option = document.createElement('option');
      option.value = '0';
      option.textContent = 'No keys configured';
      keySelect.appendChild(option);
      keySelect.disabled = true;
      runBtn.disabled = true;
      return;
    }
    keySelect.disabled = false;
    runBtn.disabled = false;
    keys.forEach((key) => {
      const option = document.createElement('option');
      option.value = String(key.index);
      option.textContent = key.preview;
      keySelect.appendChild(option);
    });
  }

  function refreshPingForm() {
    populateModels();
    populateKeys();
  }

  providerSelect.addEventListener('change', refreshPingForm);

  runBtn.addEventListener('click', async () => {
    setResult('Sending text ping…', null);
    runBtn.disabled = true;

    try {
      const response = await fetch('/admin/api/text-ping', {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: providerSelect.value,
          model: modelSelect.value,
          key_index: Number.parseInt(keySelect.value, 10) || 0
        })
      });

      const data = await response.json().catch(() => ({}));
      const line = data.message || response.statusText || 'Unknown error';
      const detail = data.actual
        ? ` Raw reply: "${data.actual}"`
        : '';
      setResult(`${line}${detail}`, Boolean(data.ok));
    } catch (err) {
      setResult(err.message || 'Request failed.', false);
    } finally {
      runBtn.disabled = keySelect.disabled;
    }
  });

  refreshPingForm();
})();
