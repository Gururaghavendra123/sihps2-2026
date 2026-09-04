let sessionId = null;
let waveformChart, fftChart, constellationChart;

const $ = (id) => document.getElementById(id);

function makeLineChart(canvasId, datasets, xLabel) {
  return new Chart($(canvasId), {
    type: "line",
    data: { datasets },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      elements: { point: { radius: 0 } },
      scales: {
        x: { type: "linear", title: { display: true, text: xLabel, color: "#2f6b34" }, ticks: { color: "#2f6b34" }, grid: { color: "#12280f" } },
        y: { ticks: { color: "#2f6b34" }, grid: { color: "#12280f" } },
      },
      plugins: { legend: { labels: { color: "#7dff7d", font: { family: "VT323", size: 14 } } } },
    },
  });
}

function initCharts() {
  waveformChart = makeLineChart(
    "waveform-chart",
    [
      { label: "I", data: [], borderColor: "#7dff7d", borderWidth: 1 },
      { label: "Q", data: [], borderColor: "#ffb020", borderWidth: 1 },
    ],
    "Sample"
  );
  fftChart = makeLineChart(
    "fft-chart",
    [{ label: "Magnitude (dB)", data: [], borderColor: "#7dff7d", borderWidth: 1 }],
    "Frequency (Hz)"
  );
  constellationChart = new Chart($("constellation-chart"), {
    type: "scatter",
    data: { datasets: [{ label: "symbols", data: [], backgroundColor: "#ffb020", pointRadius: 2 }] },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: "I", color: "#2f6b34" }, ticks: { color: "#2f6b34" }, grid: { color: "#12280f" } },
        y: { title: { display: true, text: "Q", color: "#2f6b34" }, ticks: { color: "#2f6b34" }, grid: { color: "#12280f" } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function drawSpectrogram(spec) {
  const canvas = $("spec-canvas");
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width));
  canvas.height = 220;

  const sxx = spec.sxx_db; // [freq][time]
  const nFreq = sxx.length;
  const nTime = nFreq > 0 ? sxx[0].length : 0;
  if (nFreq === 0 || nTime === 0) return;

  let min = Infinity, max = -Infinity;
  for (const row of sxx) for (const v of row) { if (v < min) min = v; if (v > max) max = v; }
  const span = max - min || 1;

  const cellW = canvas.width / nTime;
  const cellH = canvas.height / nFreq;
  for (let f = 0; f < nFreq; f++) {
    for (let t = 0; t < nTime; t++) {
      const norm = (sxx[f][t] - min) / span;
      ctx.fillStyle = viridis(norm);
      // freq axis: row 0 is lowest frequency -> draw from bottom up
      const y = canvas.height - (f + 1) * cellH;
      ctx.fillRect(t * cellW, y, cellW + 1, cellH + 1);
    }
  }
}

function viridis(t) {
  // classic Winamp spectrum-analyzer ramp: dark green -> bright green -> amber -> red peak
  const stops = [
    [8, 22, 12], [20, 90, 30], [70, 200, 60], [255, 200, 40], [255, 90, 40],
  ];
  const scaled = Math.min(0.999, Math.max(0, t)) * (stops.length - 1);
  const i = Math.floor(scaled);
  const f = scaled - i;
  const a = stops[i], b = stops[Math.min(i + 1, stops.length - 1)];
  const r = Math.round(a[0] + (b[0] - a[0]) * f);
  const g = Math.round(a[1] + (b[1] - a[1]) * f);
  const bl = Math.round(a[2] + (b[2] - a[2]) * f);
  return `rgb(${r},${g},${bl})`;
}

function renderAnalysis(data) {
  waveformChart.data.datasets[0].data = data.waveform.i.map((v, idx) => ({ x: idx, y: v }));
  waveformChart.data.datasets[1].data = data.waveform.q.map((v, idx) => ({ x: idx, y: v }));
  waveformChart.update();

  fftChart.data.datasets[0].data = data.fft.freqs.map((f, idx) => ({ x: f, y: data.fft.mag_db[idx] }));
  fftChart.update();

  constellationChart.data.datasets[0].data = data.constellation.i.map((v, idx) => ({ x: v, y: data.constellation.q[idx] }));
  constellationChart.update();

  drawSpectrogram(data.spectrogram);

  const p = data.params;
  $("params-label").textContent =
    `BW: ${(p.bandwidth_hz / 1e3).toFixed(1)} kHz | SNR: ${p.snr_db.toFixed(1)} dB | Symbol rate: ${(p.symbol_rate_hz / 1e3).toFixed(2)} kSym/s`;
}

function applyDefaults(defaults, symbolRateHz, fs) {
  // Modulation is deliberately left on "auto" — that's the point of the
  // search engine, not something to quietly pre-fill from a ground-truth
  // sidecar. sps IS derived from the just-estimated symbol rate (real DSP
  // output, dsp/analysis.estimate_symbol_rate), same as a real receiver
  // would set it — the sidecar value only overrides it when present and
  // the live estimate looks unusable (e.g. near-zero on synthetic noise).
  if (symbolRateHz > 0) {
    $("sps-input").value = Math.max(1, Math.round(fs / symbolRateHz));
  } else if (defaults && defaults.sps) {
    $("sps-input").value = defaults.sps;
  }
  if (defaults && defaults.n_payload_bits) $("payload-input").value = defaults.n_payload_bits;
}

async function handleLoadResponse(resp, label) {
  if (!resp.ok) {
    $("status-label").textContent = `load failed: ${await resp.text()}`;
    return;
  }
  const data = await resp.json();
  sessionId = data.session_id;
  renderAnalysis(data);
  applyDefaults(data.defaults, data.params.symbol_rate_hz, data.fs);
  $("status-label").textContent = `Loaded ${label} — ${data.n_samples} samples @ ${data.fs.toFixed(0)} Hz`;
  $("search-btn").disabled = false;
  $("evidence-box").textContent = "Set modulation / samples-per-symbol / payload bits, then Run Hypothesis Search.";
  $("bitstream-box").textContent = "run a search to populate";
}

async function loadSamples() {
  const resp = await fetch("/api/samples");
  const data = await resp.json();
  const sel = $("sample-select");
  for (const s of data.samples) {
    const opt = document.createElement("option");
    opt.value = s.name;
    opt.textContent = s.name;
    sel.appendChild(opt);
  }
}

$("upload-btn").addEventListener("click", () => $("file-input").click());

$("file-input").addEventListener("change", async () => {
  const file = $("file-input").files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  $("status-label").textContent = "Uploading...";
  const resp = await fetch("/api/upload", { method: "POST", body: form });
  await handleLoadResponse(resp, file.name);
});

$("sample-select").addEventListener("change", async (e) => {
  const name = e.target.value;
  if (!name) return;
  const form = new FormData();
  form.append("name", name);
  $("status-label").textContent = `Loading ${name}...`;
  const resp = await fetch("/api/load_sample", { method: "POST", body: form });
  await handleLoadResponse(resp, name);
});

$("search-btn").addEventListener("click", async () => {
  if (!sessionId) return;
  const modulation = $("mod-select").value;
  const sps = $("sps-input").value;
  const payloadBits = $("payload-input").value;

  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("modulation", modulation);
  form.append("sps", sps);
  form.append("n_payload_bits", payloadBits);

  $("search-btn").disabled = true;
  $("status-label").textContent = modulation === "auto"
    ? "Searching 6 modulations x 4 FEC x 4 interleavers (96 hypotheses)..."
    : `Searching ${modulation} x 4 FEC x 4 interleavers...`;
  try {
    const resp = await fetch("/api/search", { method: "POST", body: form });
    const data = await resp.json();
    renderEvidence(data.results, data.summary);
    renderBitstream(data.summary);
    $("status-label").textContent = `Search complete — ${data.summary.confidence}`;
  } finally {
    $("search-btn").disabled = false;
  }
});

function renderEvidence(results, summary) {
  const box = $("evidence-box");
  if (!results.length) {
    box.textContent = `HYPOTHESIS SEARCH — ${summary.confidence}\n\n${summary.message || "sync word not found"}`;
    return;
  }
  const lines = [`HYPOTHESIS SEARCH — ${summary.confidence}`, ""];
  results.slice(0, 5).forEach((r, rank) => {
    const tag = rank === 0 && summary.status === "verified" ? "  <- winner" : "";
    const modTag = r.modulation ? `${r.modulation.toUpperCase()} / ` : "";
    lines.push(`${modTag}${r.interleaver.toUpperCase()} INTERLEAVER + ${r.fec.toUpperCase().padEnd(14)} SCORE ${r.score.toFixed(0)}${tag}`);
    lines.push(`  ${r.crc_ok ? "✓" : "✗"} CRC-16: ${r.crc_ok ? "valid" : "invalid"}`);
    lines.push(`  ${r.correlation >= 0.8 ? "✓" : "✗"} Sync-word correlation: ${r.correlation.toFixed(2)}`);
    lines.push(`  ${r.ber < 0.05 ? "✓" : "✗"} Re-encode BER: ${r.ber.toFixed(3)}`);
    if (r.error) lines.push(`  ! decode error: ${r.error}`);
    lines.push("");
  });
  box.textContent = lines.join("\n");
}

function renderBitstream(summary) {
  const box = $("bitstream-box");
  if (summary.status !== "verified") {
    box.textContent = `NO VERIFIED CANDIDATE — ${summary.message || summary.status}`;
    return;
  }
  const bits = summary.payload_bits;
  const preview = bits.slice(0, 512);
  const suffix = bits.length > 512 ? "..." : "";
  const modLine = summary.modulation ? `Modulation: ${summary.modulation}   ` : "";
  box.textContent =
    `RECOVERED PAYLOAD (${bits.length} bits) — CRC VERIFIED\n` +
    `${modLine}FEC: ${summary.fec}   Interleaver: ${summary.interleaver}   Score: ${summary.score.toFixed(0)}\n\n` +
    `${preview}${suffix}`;
}

initCharts();
loadSamples();
