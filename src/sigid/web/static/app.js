/**
 * SIG-ID Tactical SIGINT Command Center Controller
 * Aesthetic: Ultra-Modern Aerospace & Defense Glassmorphism
 * Team Vertex — SIH26147 (NTRO)
 */

let sessionId = null;
let currentSignalData = null;
let waveformChart, fftChart, constellationChart, eyeChart;
let constZoomed = false;
let eyeChannelMode = "both"; // "both", "i", "q"
let activeColormap = "cyber";

let audioCtx = null;
let audioSourceNode = null;
let audioGainNode = null;
let audioPannerNode = null;
let audioFilterNode = null;
let audioDemodBuffer = null;
let isAudioPlaying = false;

let batchResultsCache = [];

const $ = (id) => document.getElementById(id);

/* ============================================================
   CHART INITIALIZATION & THEMING (TOMORRO DESIGN SYSTEM)
   ============================================================ */
const TACTICAL_COLORS = {
  cyan: "#68ef3f",       // Electric Sprout (Primary neon stroke)
  amber: "#26a200",      // Deep Verdant (Secondary accent)
  emerald: "#e7f9dd",    // Sprout Wash (Soft status highlight)
  grid: "rgba(183, 189, 165, 0.12)",
  textDim: "#7e8371",    // Lichen Sage
  textMuted: "#b7bda5",  // Pale Fern
  white: "#ffffff",      // Pure White
};

function makeTacticalLineChart(canvasId, datasets, xLabel) {
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
        x: {
          type: "linear",
          title: { display: true, text: xLabel, color: TACTICAL_COLORS.textDim, font: { family: "'Inter', sans-serif", size: 11, weight: 400 } },
          ticks: { color: TACTICAL_COLORS.textDim, font: { family: "'JetBrains Mono', monospace", size: 10 } },
          grid: { color: TACTICAL_COLORS.grid },
        },
        y: {
          ticks: { color: TACTICAL_COLORS.textDim, font: { family: "'JetBrains Mono', monospace", size: 10 } },
          grid: { color: TACTICAL_COLORS.grid },
        },
      },
      plugins: {
        legend: {
          labels: { color: TACTICAL_COLORS.textMuted, font: { family: "'Inter', sans-serif", size: 12, weight: 400 } },
        },
      },
    },
  });
}

function initCharts() {
  waveformChart = makeTacticalLineChart(
    "waveform-chart",
    [
      { label: "In-Phase (I)", data: [], borderColor: TACTICAL_COLORS.cyan, borderWidth: 1.4 },
      { label: "Quadrature (Q)", data: [], borderColor: TACTICAL_COLORS.amber, borderWidth: 1.4 },
    ],
    "Sample Index"
  );

  fftChart = makeTacticalLineChart(
    "fft-chart",
    [{ label: "Magnitude (dB)", data: [], borderColor: TACTICAL_COLORS.cyan, borderWidth: 1.4 }],
    "Frequency (Hz)"
  );

  constellationChart = new Chart($("constellation-chart"), {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "I/Q Symbols",
          data: [],
          backgroundColor: TACTICAL_COLORS.cyan,
          pointRadius: 2.4,
          pointHoverRadius: 4,
        },
      ],
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: "I (In-Phase)", color: TACTICAL_COLORS.textDim, font: { family: "'Inter', sans-serif", size: 11, weight: 400 } },
          ticks: { color: TACTICAL_COLORS.textDim, font: { family: "'JetBrains Mono', monospace", size: 10 } },
          grid: { color: TACTICAL_COLORS.grid },
        },
        y: {
          title: { display: true, text: "Q (Quadrature)", color: TACTICAL_COLORS.textDim, font: { family: "'Inter', sans-serif", size: 11, weight: 400 } },
          ticks: { color: TACTICAL_COLORS.textDim, font: { family: "'JetBrains Mono', monospace", size: 10 } },
          grid: { color: TACTICAL_COLORS.grid },
        },
      },
      plugins: { legend: { display: false } },
    },
  });

  // Eye Diagram Line Chart
  eyeChart = new Chart($("eye-chart"), {
    type: "line",
    data: { datasets: [] },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      elements: { point: { radius: 0 } },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "Symbol Period (T_sym)", color: TACTICAL_COLORS.textDim, font: { family: "'Inter', sans-serif", size: 11, weight: 400 } },
          ticks: { color: TACTICAL_COLORS.textDim, font: { family: "'JetBrains Mono', monospace", size: 10 } },
          grid: { color: TACTICAL_COLORS.grid },
        },
        y: {
          ticks: { color: TACTICAL_COLORS.textDim, font: { family: "'JetBrains Mono', monospace", size: 10 } },
          grid: { color: TACTICAL_COLORS.grid },
        },
      },
      plugins: { legend: { display: false } },
    },
  });
}

/* ============================================================
   WATERFALL SPECTROGRAM COLORMAPS
   ============================================================ */
function getPaletteRamp(name, t) {
  const val = Math.min(0.999, Math.max(0, t));
  if (name === "turbo") {
    const stops = [
      [30, 0, 100],
      [0, 100, 255],
      [0, 255, 180],
      [255, 220, 0],
      [255, 30, 0],
    ];
    return interpolateStops(stops, val);
  } else if (name === "viridis") {
    const stops = [
      [68, 1, 84],
      [59, 82, 139],
      [33, 145, 140],
      [94, 201, 98],
      [253, 231, 37],
    ];
    return interpolateStops(stops, val);
  } else if (name === "inferno") {
    const stops = [
      [0, 0, 4],
      [87, 16, 110],
      [187, 55, 84],
      [249, 142, 9],
      [252, 255, 164],
    ];
    return interpolateStops(stops, val);
  } else if (name === "matrix") {
    const stops = [
      [0, 10, 0],
      [0, 80, 20],
      [0, 200, 60],
      [50, 255, 120],
      [220, 255, 230],
    ];
    return interpolateStops(stops, val);
  } else if (name === "mono") {
    const v = Math.round(val * 255);
    return `rgb(${v}, ${v}, ${v})`;
  } else {
    // Cyber Cyan Default
    const stops = [
      [4, 7, 12],
      [0, 140, 210],
      [0, 240, 255],
      [52, 211, 153],
      [251, 191, 36],
    ];
    return interpolateStops(stops, val);
  }
}

function interpolateStops(stops, t) {
  const scaled = t * (stops.length - 1);
  const i = Math.floor(scaled);
  const f = scaled - i;
  const a = stops[i], b = stops[Math.min(i + 1, stops.length - 1)];
  const r = Math.round(a[0] + (b[0] - a[0]) * f);
  const g = Math.round(a[1] + (b[1] - a[1]) * f);
  const bl = Math.round(a[2] + (b[2] - a[2]) * f);
  return `rgb(${r}, ${g}, ${bl})`;
}

function drawSpectrogram(spec) {
  const canvas = $("spec-canvas");
  if (!canvas || !spec) return;
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width));
  canvas.height = 230;

  const sxx = spec.sxx_db;
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
      ctx.fillStyle = getPaletteRamp(activeColormap, norm);
      const y = canvas.height - (f + 1) * cellH;
      ctx.fillRect(t * cellW, y, cellW + 1, cellH + 1);
    }
  }
}

/* ============================================================
   EYE DIAGRAM RENDERER
   ============================================================ */
function updateEyeDiagram(eyeData) {
  if (!eyeChart || !eyeData || !eyeData.t || eyeData.t.length === 0) return;
  const datasets = [];
  const t = eyeData.t;

  if (eyeChannelMode === "both" || eyeChannelMode === "i") {
    for (let i = 0; i < (eyeData.traces_i || []).length; i++) {
      const pts = eyeData.traces_i[i].map((val, idx) => ({ x: t[idx], y: val }));
      datasets.push({
        label: i === 0 ? "I-Eye" : "",
        data: pts,
        borderColor: "rgba(0, 240, 255, 0.22)",
        borderWidth: 1.0,
      });
    }
  }

  if (eyeChannelMode === "both" || eyeChannelMode === "q") {
    for (let i = 0; i < (eyeData.traces_q || []).length; i++) {
      const pts = eyeData.traces_q[i].map((val, idx) => ({ x: t[idx], y: val }));
      datasets.push({
        label: i === 0 ? "Q-Eye" : "",
        data: pts,
        borderColor: "rgba(245, 158, 11, 0.22)",
        borderWidth: 1.0,
      });
    }
  }

  eyeChart.data.datasets = datasets;
  eyeChart.update();
}

/* ============================================================
   AUDIO DEMODULATOR & FREQUENCY TUNER (Web Audio API)
   ============================================================ */
function initAudio() {
  if (!audioCtx) {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AudioContext();
    audioGainNode = audioCtx.createGain();
    audioPannerNode = audioCtx.createStereoPanner ? audioCtx.createStereoPanner() : null;
    audioFilterNode = audioCtx.createBiquadFilter();

    audioFilterNode.type = "lowpass";
    audioFilterNode.frequency.value = parseFloat($("audio-filter-slider").value) || 4000;

    if (audioPannerNode) {
      audioFilterNode.connect(audioGainNode);
      audioGainNode.connect(audioPannerNode);
      audioPannerNode.connect(audioCtx.destination);
    } else {
      audioFilterNode.connect(audioGainNode);
      audioGainNode.connect(audioCtx.destination);
    }
  }
  if (audioCtx.state === "suspended") {
    audioCtx.resume();
  }
}

function buildDemodAudioBuffer(iqI, iqQ, fs, mode) {
  if (!audioCtx) return null;
  const n = iqI.length;
  const buffer = audioCtx.createBuffer(2, n, Math.min(fs, 48000));
  const ch0 = buffer.getChannelData(0);
  const ch1 = buffer.getChannelData(1);

  if (mode === "am") {
    for (let i = 0; i < n; i++) {
      const env = Math.sqrt(iqI[i] * iqI[i] + iqQ[i] * iqQ[i]);
      ch0[i] = env * 0.8;
      ch1[i] = env * 0.8;
    }
  } else if (mode === "fm") {
    let lastPhase = Math.atan2(iqQ[0], iqI[0]);
    for (let i = 1; i < n; i++) {
      const phase = Math.atan2(iqQ[i], iqI[i]);
      let diff = phase - lastPhase;
      while (diff > Math.PI) diff -= 2 * Math.PI;
      while (diff < -Math.PI) diff += 2 * Math.PI;
      ch0[i] = diff / Math.PI;
      ch1[i] = diff / Math.PI;
      lastPhase = phase;
    }
  } else {
    for (let i = 0; i < n; i++) {
      ch0[i] = iqI[i] * 0.7;
      ch1[i] = iqQ[i] * 0.7;
    }
  }
  return buffer;
}

function playAudio() {
  if (!currentSignalData) return;
  initAudio();
  stopAudio();

  const mode = $("audio-demod-mode").value;
  const wf = currentSignalData.waveform;
  if (!wf || !wf.i || wf.i.length === 0) return;

  audioDemodBuffer = buildDemodAudioBuffer(wf.i, wf.q, currentSignalData.fs || 44100, mode);
  if (!audioDemodBuffer) return;

  audioSourceNode = audioCtx.createBufferSource();
  audioSourceNode.buffer = audioDemodBuffer;
  audioSourceNode.loop = true;
  audioSourceNode.connect(audioFilterNode);

  updateVolume();

  audioSourceNode.start(0);
  isAudioPlaying = true;
}

function pauseAudio() {
  if (audioCtx && isAudioPlaying) {
    if (audioCtx.state === "running") {
      audioCtx.suspend();
    } else {
      audioCtx.resume();
    }
  }
}

function stopAudio() {
  if (audioSourceNode) {
    try { audioSourceNode.stop(0); } catch (e) {}
    audioSourceNode.disconnect();
    audioSourceNode = null;
  }
  isAudioPlaying = false;
}

function updateVolume() {
  if (!audioGainNode) return;
  const val = parseFloat($("vol-slider").value) / 100;
  audioGainNode.gain.setValueAtTime(val, audioCtx.currentTime);
  $("vol-val").textContent = `${Math.round(val * 100)}%`;
}

/* ============================================================
   SIGNAL ANALYSIS STATE & UI UPDATE
   ============================================================ */
function updateDashboardState(data) {
  currentSignalData = data;
  sessionId = data.session_id;

  // Update Telemetry Strip
  if (data.params) {
    $("lcd-bw").textContent = `${(data.params.bandwidth_hz / 1000).toFixed(1)} kHz`;
    $("lcd-snr").textContent = `${data.params.snr_db.toFixed(1)} dB`;
    $("lcd-symrate").textContent = `${(data.params.symbol_rate_hz / 1000).toFixed(1)} kSym/s`;
  }
  if (data.features && data.features.cfo_estimated_hz !== undefined) {
    const cfo = data.features.cfo_estimated_hz;
    $("lcd-cfo").textContent = `${cfo > 0 ? "+" : ""}${cfo.toFixed(1)} Hz`;
  }

  $("badge-rate").textContent = `${Math.round((data.fs || 200000) / 1000)}K FS`;
  $("lcd-status-text").textContent = "LOADED";
  $("search-btn").disabled = false;

  // Top Classification HUD
  if (data.features && data.features.top_modulations && data.features.top_modulations.length > 0) {
    const top = data.features.top_modulations[0];
    $("hud-top-mod").textContent = top.mod.toUpperCase();
    $("hud-top-score").textContent = `SCORE: ${top.score.toFixed(3)}`;
  }

  // Waveform
  if (waveformChart && data.waveform) {
    waveformChart.data.datasets[0].data = data.waveform.i.map((v, i) => ({ x: i, y: v }));
    waveformChart.data.datasets[1].data = data.waveform.q.map((v, i) => ({ x: i, y: v }));
    waveformChart.update();
  }

  // FFT
  if (fftChart && data.fft) {
    fftChart.data.datasets[0].data = data.fft.freqs.map((f, i) => ({ x: f, y: data.fft.mag_db[i] }));
    fftChart.update();
  }

  // Constellation
  if (constellationChart && data.constellation) {
    constellationChart.data.datasets[0].data = data.constellation.i.map((v, i) => ({ x: v, y: data.constellation.q[i] }));
    if (!constZoomed) {
      constellationChart.options.scales.x.min = -2.2;
      constellationChart.options.scales.x.max = 2.2;
      constellationChart.options.scales.y.min = -2.2;
      constellationChart.options.scales.y.max = 2.2;
    }
    constellationChart.update();
  }

  // Waterfall
  if (data.spectrogram) {
    drawSpectrogram(data.spectrogram);
  }

  // Eye Diagram
  if (data.eye_diagram) {
    updateEyeDiagram(data.eye_diagram);
  }

  // Cumulants Deck
  if (data.features) {
    $("val-c20").textContent = data.features.C20.toFixed(5);
    $("val-c40").textContent = data.features.C40.toFixed(5);
    $("val-c42").textContent = data.features.C42.toFixed(5);
    $("val-envvar").textContent = data.features.envelope_var.toFixed(5);
    $("val-kurtosis").textContent = data.features.freq_kurtosis.toFixed(5);
    $("val-cfo-est").textContent = `${data.features.cfo_estimated_hz.toFixed(1)} Hz`;

    // Ranking Bars
    const topMods = data.features.top_modulations || [];
    let barHtml = "";
    for (const m of topMods) {
      const pct = Math.max(5, Math.min(100, Math.round(m.score * 100)));
      barHtml += `
        <div class="ranking-item">
          <span class="mod-tag">${m.mod.toUpperCase()}</span>
          <div class="bar-track"><div class="bar-fill" style="width: ${pct}%;"></div></div>
          <span class="bar-score">${m.score.toFixed(3)}</span>
        </div>`;
    }
    $("mod-rankings-bars").innerHTML = barHtml || "<div style='color:var(--text-dim);'>Extracting features...</div>";
  }

  // Defaults Sync
  if (data.defaults) {
    if (data.defaults.sps) {
      $("sps-input").value = data.defaults.sps;
      $("hud-sps-tag").textContent = data.defaults.sps;
    }
    if (data.defaults.n_payload_bits) $("payload-input").value = data.defaults.n_payload_bits;
  }
}

/* ============================================================
   CFO ZERO COMPENSATION
   ============================================================ */
async function compensateCFO() {
  if (!sessionId || !currentSignalData || !currentSignalData.features) return;
  const cfoEst = currentSignalData.features.cfo_estimated_hz || 0;
  if (Math.abs(cfoEst) < 0.1) {
    alert("Carrier frequency offset is already within ±0.1 Hz!");
    return;
  }

  const formData = new FormData();
  formData.append("session_id", sessionId);
  formData.append("cfo_hz", -cfoEst);

  $("lcd-status-text").textContent = "ZEROING...";
  try {
    const res = await fetch("/api/cfo_compensate", { method: "POST", body: formData });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    updateDashboardState(data);
    $("lcd-status-text").textContent = "ZEROED";
  } catch (err) {
    alert(`CFO compensation failed: ${err.message}`);
    $("lcd-status-text").textContent = "ERROR";
  }
}

/* ============================================================
   EQUALIZER & DSP FILTER TUNER
   ============================================================ */
async function applyDSPFilter() {
  if (!sessionId) {
    alert("Load a signal recording first!");
    return;
  }

  const mode = $("filter-type-select").value;
  const faders = document.querySelectorAll(".eq-vert-slider");
  const gains = Array.from(faders).map((f) => f.value).join(",");

  let cutoffLow = 20000;
  let cutoffHigh = 60000;
  if (mode === "lowpass") cutoffLow = 20000;
  else if (mode === "highpass") cutoffLow = 5000;
  else if (mode === "bandpass") { cutoffLow = 10000; cutoffHigh = 40000; }

  const formData = new FormData();
  formData.append("session_id", sessionId);
  formData.append("filter_type", mode);
  formData.append("cutoff_hz", cutoffLow);
  formData.append("cutoff_high_hz", cutoffHigh);
  formData.append("eq_gains", gains);

  $("lcd-status-text").textContent = "FILTERING...";
  try {
    const res = await fetch("/api/filter", { method: "POST", body: formData });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    updateDashboardState(data);
    $("lcd-status-text").textContent = "FILTERED";
  } catch (err) {
    alert(`Filter error: ${err.message}`);
    $("lcd-status-text").textContent = "ERROR";
  }
}

async function resetDSPFilter() {
  if (!sessionId) return;
  const formData = new FormData();
  formData.append("session_id", sessionId);

  document.querySelectorAll(".eq-vert-slider").forEach((f) => { f.value = 0; });

  $("lcd-status-text").textContent = "RESETTING...";
  try {
    const res = await fetch("/api/reset_filter", { method: "POST", body: formData });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    updateDashboardState(data);
    $("lcd-status-text").textContent = "RAW LOADED";
  } catch (err) {
    alert(`Reset error: ${err.message}`);
    $("lcd-status-text").textContent = "ERROR";
  }
}

/* ============================================================
   HYPOTHESIS SEARCH & VERIFICATION
   ============================================================ */
async function runSearch() {
  if (!sessionId) {
    alert("Load a signal recording first!");
    return;
  }

  const engineMode = document.querySelector('input[name="engine-mode"]:checked').value;
  const mod = $("mod-select").value;
  const sps = parseInt($("sps-input").value) || 4;
  const payloadBits = parseInt($("payload-input").value) || 256;

  $("search-btn").disabled = true;
  $("search-btn").innerHTML = "EXECUTING...";
  $("lcd-status-text").textContent = "SEARCHING";
  $("badge-engine").textContent = engineMode.toUpperCase();

  const formData = new FormData();
  formData.append("session_id", sessionId);
  formData.append("modulation", mod);
  formData.append("sps", sps);
  formData.append("n_payload_bits", payloadBits);
  formData.append("mode", engineMode);

  try {
    const res = await fetch("/api/search", { method: "POST", body: formData });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    renderSearchResults(data);
    $("lcd-status-text").textContent = data.summary && data.summary.status === "verified" ? "VERIFIED" : "UNVERIFIED";
  } catch (err) {
    alert(`Search execution failed: ${err.message}`);
    $("lcd-status-text").textContent = "ERROR";
  } finally {
    $("search-btn").disabled = false;
    $("search-btn").innerHTML = "EXECUTE SEARCH";
  }
}

function renderSearchResults(data) {
  const summary = data.summary || {};
  const results = data.results || [];
  const tbody = $("playlist-table-body");

  if (summary.status === "verified") {
    $("badge-verified").textContent = "CRC VERIFIED";
    $("badge-verified").style.color = "var(--emerald-verified)";
    $("badge-crc").textContent = "CRC PASS";
    $("badge-crc").style.color = "var(--emerald-verified)";
  } else {
    $("badge-verified").textContent = "UNVERIFIED";
    $("badge-verified").style.color = "var(--rose-threat)";
    $("badge-crc").textContent = "CRC FAIL";
    $("badge-crc").style.color = "var(--rose-threat)";
  }

  let html = "";
  results.forEach((r, idx) => {
    const isWin = summary.status === "verified" &&
                  r.modulation === summary.modulation &&
                  r.fec === summary.fec &&
                  r.interleaver === summary.interleaver;
    
    const rowCls = isWin ? "winner-row" : "";
    const crcTag = r.crc_ok 
      ? `<span class="tactical-status-badge pass">PASS</span>` 
      : `<span class="tactical-status-badge fail">FAIL</span>`;
    const verdict = r.crc_ok 
      ? `<span style="color:var(--emerald-verified); font-weight:700;">VERIFIED</span>` 
      : `<span style="color:var(--text-dim);">REJECT</span>`;

    html += `
      <tr class="${rowCls}">
        <td>${idx + 1}</td>
        <td><b>${(r.modulation || "").toUpperCase()}</b></td>
        <td>${(r.fec || "").toUpperCase()}</td>
        <td>${(r.interleaver || "").toUpperCase()}</td>
        <td>${crcTag}</td>
        <td>${(r.correlation || 0).toFixed(4)}</td>
        <td>${(r.ber !== undefined ? r.ber.toFixed(4) : "1.0000")}</td>
        <td><b>${(r.score || 0).toFixed(2)}</b></td>
        <td>${verdict}</td>
      </tr>`;
  });
  tbody.innerHTML = html;

  renderBitstreamExploitation(summary);

  $("evidence-box").textContent = JSON.stringify({ summary, top_candidates: results.slice(0, 5) }, null, 2);

  if (data.adaptive_summary) {
    $("telemetry-win").style.display = "block";
    $("telemetry-box").textContent = JSON.stringify({
      adaptive_summary: data.adaptive_summary,
      telemetry: data.telemetry,
    }, null, 2);
  }
}

/* ============================================================
   BITSTREAM HEX & ASCII & BER DIFF EXPLOITATION
   ============================================================ */
function renderBitstreamExploitation(summary) {
  const payloadBits = summary.payload_bits || "";
  const hexBox = $("hex-dump-box");
  const rawBitsBox = $("raw-bits-box");
  const diffGrid = $("bit-diff-grid");
  const diffStrip = $("diff-summary-strip");

  if (!payloadBits) {
    hexBox.textContent = "No recovered payload data. Signal unverified.";
    rawBitsBox.textContent = "No payload recovered.";
    diffGrid.textContent = "No payload recovered.";
    return;
  }

  rawBitsBox.textContent = payloadBits.match(/.{1,8}/g).join(" ");

  const nBytes = Math.floor(payloadBits.length / 8);
  const bytes = [];
  for (let i = 0; i < nBytes; i++) {
    bytes.push(parseInt(payloadBits.substr(i * 8, 8), 2));
  }

  let hexLines = [];
  for (let offset = 0; offset < bytes.length; offset += 16) {
    const chunk = bytes.slice(offset, offset + 16);
    const hexPart = chunk.map((b) => b.toString(16).padStart(2, "0").toUpperCase()).join(" ").padEnd(48, " ");
    const asciiPart = chunk.map((b) => (b >= 32 && b <= 126 ? String.fromCharCode(b) : ".")).join("");
    hexLines.push(`${offset.toString(16).padStart(4, "0").toUpperCase()}  ${hexPart}  |${asciiPart}|`);
  }
  hexBox.textContent = hexLines.join("\n");

  const gt = currentSignalData ? currentSignalData.ground_truth : null;
  if (gt && gt.payload_bits && gt.payload_bits.length > 0) {
    const gtBits = gt.payload_bits;
    const len = Math.min(gtBits.length, payloadBits.length);
    let flips = 0;
    let chipsHtml = "";

    for (let i = 0; i < len; i++) {
      const recBit = parseInt(payloadBits[i]);
      const gtBit = gtBits[i];
      if (recBit !== gtBit) {
        flips++;
        chipsHtml += `<span class="bit-chip flip" title="Bit ${i}: Got ${recBit}, Expected ${gtBit}">${recBit}</span>`;
      } else {
        chipsHtml += `<span class="bit-chip match">${recBit}</span>`;
      }
    }

    const ber = (flips / len).toFixed(4);
    diffStrip.innerHTML = `BER: <b>${ber}</b> &bull; BIT ERRORS: <b>${flips}/${len}</b> &bull; CRC-16: <b>${summary.status === "verified" ? "PASS" : "FAIL"}</b>`;
    diffGrid.innerHTML = chipsHtml;
  } else {
    diffStrip.innerHTML = "GROUND TRUTH: Not loaded (synthetic payload verified by CRC-16 checksum)";
    diffGrid.innerHTML = payloadBits.split("").map((b) => `<span class="bit-chip match">${b}</span>`).join("");
  }
}

/* ============================================================
   SYNTHESIS LAB MODAL & INJECTOR
   ============================================================ */
function initSynthesisLab() {
  const modal = $("synth-modal");
  $("btn-open-synth").addEventListener("click", () => { modal.style.display = "flex"; });
  $("btn-close-synth").addEventListener("click", () => { modal.style.display = "none"; });
  $("btn-cancel-synth").addEventListener("click", () => { modal.style.display = "none"; });

  $("synth-snr").addEventListener("input", (e) => { $("synth-snr-val").textContent = `${e.target.value} dB`; });
  $("synth-cfo").addEventListener("input", (e) => { $("synth-cfo-val").textContent = `${e.target.value} Hz`; });
  $("synth-mp-delay").addEventListener("input", (e) => { $("synth-mp-delay-val").textContent = e.target.value; });
  $("synth-mp-att").addEventListener("input", (e) => { $("synth-mp-att-val").textContent = parseFloat(e.target.value).toFixed(2); });

  document.querySelectorAll(".synth-preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      $("synth-mod").value = btn.dataset.mod;
      $("synth-fec").value = btn.dataset.fec;
      $("synth-intl").value = btn.dataset.intl;
      $("synth-snr").value = btn.dataset.snr;
      $("synth-snr-val").textContent = `${btn.dataset.snr} dB`;
      $("synth-cfo").value = btn.dataset.cfo;
      $("synth-cfo-val").textContent = `${btn.dataset.cfo} Hz`;
      $("synth-mp-delay").value = btn.dataset.delay;
      $("synth-mp-delay-val").textContent = btn.dataset.delay;
      $("synth-mp-att").value = btn.dataset.att;
      $("synth-mp-att-val").textContent = parseFloat(btn.dataset.att).toFixed(2);
    });
  });

  $("btn-do-synth").addEventListener("click", async () => {
    const formData = new FormData();
    formData.append("modulation", $("synth-mod").value);
    formData.append("fec", $("synth-fec").value);
    formData.append("interleaver", $("synth-intl").value);
    formData.append("sps", $("synth-sps").value);
    formData.append("snr_db", $("synth-snr").value);
    formData.append("freq_offset_hz", $("synth-cfo").value);
    formData.append("multipath_delay", $("synth-mp-delay").value);
    formData.append("multipath_attenuation", $("synth-mp-att").value);
    formData.append("custom_text", $("synth-custom-text").value);

    $("btn-do-synth").disabled = true;
    $("btn-do-synth").textContent = "SYNTHESIZING...";

    try {
      const res = await fetch("/api/synthesize", { method: "POST", body: formData });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      updateDashboardState(data);
      modal.style.display = "none";
      alert(`Synthetic signal successfully generated and injected into Command Deck!\nSession ID: ${data.session_id.substring(0, 8)}`);
    } catch (err) {
      alert(`Synthesis failed: ${err.message}`);
    } finally {
      $("btn-do-synth").disabled = false;
      $("btn-do-synth").textContent = "INJECT INTO COMMAND DECK";
    }
  });
}

/* ============================================================
   BATCH BENCHMARK MATRIX SUITE
   ============================================================ */
function initBatchBenchmark() {
  const modal = $("batch-modal");
  $("btn-open-batch").addEventListener("click", () => { modal.style.display = "flex"; });
  $("btn-close-batch").addEventListener("click", () => { modal.style.display = "none"; });

  $("btn-run-batch-matrix").addEventListener("click", async () => {
    const engine = document.querySelector('input[name="batch-engine"]:checked').value;
    const btn = $("btn-run-batch-matrix");
    btn.disabled = true;
    btn.textContent = "EXECUTING...";

    const formData = new FormData();
    formData.append("mode", engine);

    try {
      const res = await fetch("/api/batch_search", { method: "POST", body: formData });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      batchResultsCache = data.batch_results || [];

      renderBatchMatrix(batchResultsCache);
      $("btn-export-batch-csv").disabled = false;
    } catch (err) {
      alert(`Batch benchmark failed: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = "RUN ALL BENCHMARKS";
    }
  });

  $("btn-export-batch-csv").addEventListener("click", () => {
    if (!batchResultsCache || batchResultsCache.length === 0) return;
    const lines = ["sample,status,confidence,winner_mod,winner_fec,winner_interleaver,score,decodes_run,reduction_pct"];
    for (const r of batchResultsCache) {
      lines.push(`${r.sample},${r.status},${r.confidence},${r.winner_mod || ""},${r.winner_fec || ""},${r.winner_interleaver || ""},${r.score || 0},${r.decodes_run},${r.reduction_pct}%`);
    }
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `batch_benchmark_results.csv`;
    a.click();
  });
}

function renderBatchMatrix(records) {
  const tbody = $("batch-results-tbody");
  $("batch-stats-summary").style.display = "grid";

  let verifiedCount = 0;
  let totalReduction = 0;
  let html = "";

  records.forEach((r) => {
    if (r.status === "verified") verifiedCount++;
    totalReduction += parseFloat(r.reduction_pct) || 0;

    const statusTag = r.status === "verified"
      ? `<span class="tactical-status-badge pass">VERIFIED</span>`
      : `<span class="tactical-status-badge fail">UNVERIFIED</span>`;

    html += `
      <tr>
        <td><b>${r.sample}</b></td>
        <td>${statusTag}</td>
        <td>${r.confidence || "N/A"}</td>
        <td><b>${(r.winner_mod || "NONE").toUpperCase()}</b></td>
        <td>${(r.winner_fec || "NONE").toUpperCase()}</td>
        <td>${(r.winner_interleaver || "NONE").toUpperCase()}</td>
        <td>${(r.score || 0).toFixed(1)}</td>
        <td><code>${r.decodes_run}</code></td>
        <td><b>${r.reduction_pct}%</b></td>
      </tr>`;
  });
  tbody.innerHTML = html;

  $("stat-total-signals").textContent = records.length;
  $("stat-pass-rate").textContent = records.length > 0 ? `${Math.round((verifiedCount / records.length) * 100)}%` : "0%";
  const avgRed = records.length > 0 ? Math.round(totalReduction / records.length) : 0;
  $("stat-avg-reduction").textContent = `${avgRed}%`;
  const speedup = avgRed > 0 ? (100 / (100 - avgRed)).toFixed(1) : "1.0";
  $("stat-avg-speedup").textContent = `${speedup}x`;
}

/* ============================================================
   EXPORT ACTIONS & DOSSIER
   ============================================================ */
function initExports() {
  function downloadExport(fmt) {
    if (!sessionId) {
      alert("Load or synthesize a signal recording first!");
      return;
    }
    window.open(`/api/export/${sessionId}/${fmt}`, "_blank");
  }

  $("exp-iq").addEventListener("click", (e) => { e.preventDefault(); downloadExport("iq"); });
  $("exp-wav").addEventListener("click", (e) => { e.preventDefault(); downloadExport("wav"); });
  $("exp-json").addEventListener("click", (e) => { e.preventDefault(); downloadExport("json"); });
  $("exp-csv").addEventListener("click", (e) => { e.preventDefault(); downloadExport("csv"); });
  $("exp-dossier").addEventListener("click", (e) => { e.preventDefault(); downloadExport("dossier"); });
  $("btn-open-dossier").addEventListener("click", () => { downloadExport("dossier"); });
}

/* ============================================================
   SAMPLE LOADING & FILE UPLOADS
   ============================================================ */
async function loadSamplesList() {
  try {
    const res = await fetch("/api/samples");
    const data = await res.json();
    const select = $("sample-select");
    select.innerHTML = '<option value="">-- BENCHMARK SAMPLES --</option>';
    (data.samples || []).forEach((s) => {
      const opt = document.createElement("option");
      opt.value = s.name;
      opt.textContent = s.name;
      select.appendChild(opt);
    });
  } catch (e) {
    console.error("Failed to load sample list", e);
  }
}

async function loadSample(name) {
  if (!name) return;
  const formData = new FormData();
  formData.append("name", name);
  $("lcd-status-text").textContent = "LOADING...";

  try {
    const res = await fetch("/api/load_sample", { method: "POST", body: formData });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    updateDashboardState(data);
  } catch (err) {
    alert(`Failed to load sample: ${err.message}`);
    $("lcd-status-text").textContent = "ERROR";
  }
}

/* ============================================================
   BOOTSTRAP EVENT HANDLERS
   ============================================================ */
window.addEventListener("DOMContentLoaded", () => {
  initCharts();
  initSynthesisLab();
  initBatchBenchmark();
  initExports();
  loadSamplesList();

  // Load sample dropdown
  $("sample-select").addEventListener("change", (e) => {
    loadSample(e.target.value);
  });

  // Upload file
  $("upload-btn").addEventListener("click", () => $("file-input").click());
  $("file-input").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    $("lcd-status-text").textContent = "UPLOADING...";

    try {
      const res = await fetch("/api/upload", { method: "POST", body: formData });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      updateDashboardState(data);
    } catch (err) {
      alert(`Upload failed: ${err.message}`);
      $("lcd-status-text").textContent = "ERROR";
    }
  });

  // Engine Radio Toggle
  const radioLabels = document.querySelectorAll(".tactical-radio-group .tactical-radio-label");
  document.querySelectorAll('input[name="engine-mode"]').forEach((input) => {
    input.addEventListener("change", () => {
      radioLabels.forEach((l) => l.classList.remove("active"));
      input.closest(".tactical-radio-label").classList.add("active");
    });
  });

  // Transport audio buttons
  $("btn-play").addEventListener("click", playAudio);
  $("btn-pause").addEventListener("click", pauseAudio);
  $("btn-stop").addEventListener("click", stopAudio);
  $("btn-prev").addEventListener("click", () => { stopAudio(); playAudio(); });
  $("btn-next").addEventListener("click", () => {
    document.querySelector('input[name="engine-mode"][value="adaptive"]').checked = true;
    radioLabels.forEach((l) => l.classList.remove("active"));
    $("label-engine-adapt").classList.add("active");
    runSearch();
  });

  // Volume & Filter Sliders
  $("vol-slider").addEventListener("input", updateVolume);
  $("audio-filter-slider").addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    $("audio-filter-val").textContent = `${Math.round(val / 100) / 10} kHz`;
    if (audioFilterNode && audioCtx) {
      audioFilterNode.frequency.setValueAtTime(val, audioCtx.currentTime);
    }
  });

  // CFO Zero
  $("btn-zero-cfo").addEventListener("click", compensateCFO);

  // Equalizer DSP Filter actions
  $("btn-apply-eq").addEventListener("click", applyDSPFilter);
  $("btn-reset-eq").addEventListener("click", resetDSPFilter);

  // Search Action
  $("search-btn").addEventListener("click", runSearch);

  // Spectrogram Colormap Selector
  $("spec-colormap-select").addEventListener("change", (e) => {
    activeColormap = e.target.value;
    if (currentSignalData && currentSignalData.spectrogram) {
      drawSpectrogram(currentSignalData.spectrogram);
    }
  });

  // Constellation Zoom Toggle
  $("btn-zoom-const").addEventListener("click", () => {
    constZoomed = !constZoomed;
    $("btn-zoom-const").textContent = constZoomed ? "RESET" : "ZOOM";
    if (constellationChart) {
      if (constZoomed) {
        delete constellationChart.options.scales.x.min;
        delete constellationChart.options.scales.x.max;
        delete constellationChart.options.scales.y.min;
        delete constellationChart.options.scales.y.max;
      } else {
        constellationChart.options.scales.x.min = -2.2;
        constellationChart.options.scales.x.max = 2.2;
        constellationChart.options.scales.y.min = -2.2;
        constellationChart.options.scales.y.max = 2.2;
      }
      constellationChart.update();
    }
  });

  // Eye Diagram Channel Toggle
  $("btn-eye-toggle").addEventListener("click", () => {
    if (eyeChannelMode === "both") eyeChannelMode = "i";
    else if (eyeChannelMode === "i") eyeChannelMode = "q";
    else eyeChannelMode = "both";

    $("btn-eye-toggle").textContent = `CHANNEL: ${eyeChannelMode.toUpperCase()}`;
    if (currentSignalData && currentSignalData.eye_diagram) {
      updateEyeDiagram(currentSignalData.eye_diagram);
    }
  });

  // Bitstream Tab Switching
  const bitTabs = [
    { btn: "tab-btn-hex", view: "view-hex" },
    { btn: "tab-btn-diff", view: "view-diff" },
    { btn: "tab-btn-bits", view: "view-bits" },
    { btn: "tab-btn-evidence", view: "view-evidence" },
  ];
  bitTabs.forEach((t) => {
    $(t.btn).addEventListener("click", () => {
      bitTabs.forEach((o) => {
        $(o.btn).classList.remove("active");
        $(o.view).classList.remove("active");
      });
      $(t.btn).classList.add("active");
      $(t.view).classList.add("active");
    });
  });

  // Copy Bitstream
  $("btn-copy-bits").addEventListener("click", () => {
    const activeView = document.querySelector(".bitstream-view-panel.active");
    if (!activeView) return;
    const text = activeView.innerText;
    navigator.clipboard.writeText(text).then(() => {
      const orig = $("btn-copy-bits").textContent;
      $("btn-copy-bits").textContent = "COPIED";
      setTimeout(() => { $("btn-copy-bits").textContent = orig; }, 1500);
    });
  });

  // Candidate Filter
  $("filter-candidates").addEventListener("input", (e) => {
    const val = e.target.value.toLowerCase();
    const rows = document.querySelectorAll("#playlist-table-body tr");
    rows.forEach((r) => {
      const txt = r.textContent.toLowerCase();
      r.style.display = txt.includes(val) ? "" : "none";
    });
  });
});
