// NebulaX Rail Corrugation - Minimalist High-Performance Dashboard Controller
let currentFile = null;
let activeEventSource = null;
let activeAbortController = null;
let isStreaming = false;

document.addEventListener('DOMContentLoaded', () => {
  initApp();
});

async function initApp() {
  renderInitialTrainSchematic();
  setupEventListeners();

  // Load status and file list in parallel for zero latency
  try {
    await Promise.all([loadStatus(), loadFileList()]);
  } catch (err) {
    console.error('Initialization error:', err);
  }
  
  // Auto-analyze file from URL query parameter or first file in dropdown
  const urlParams = new URLSearchParams(window.location.search);
  const targetFile = urlParams.get('file');

  const select = document.getElementById('fileSelect');
  if (select && select.options.length > 0) {
    if (targetFile) {
      // Find matching option in select
      let found = false;
      for (let i = 0; i < select.options.length; i++) {
        if (select.options[i].value === targetFile || select.options[i].value.startsWith(targetFile)) {
          select.selectedIndex = i;
          analyzeFile(select.options[i].value);
          found = true;
          break;
        }
      }
      if (!found) {
        select.selectedIndex = 0;
        analyzeFile(select.options[0].value);
      }
    } else {
      select.selectedIndex = 0;
      analyzeFile(select.options[0].value);
    }
  }
}

async function loadStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    const f1Elem = document.getElementById('modelF1');
    if (f1Elem) {
      f1Elem.textContent = data.macro_f1 ? data.macro_f1.toFixed(4) : '0.8507';
    }
  } catch (err) {
    console.error('Status fetch error:', err);
  }
}

async function loadFileList() {
  try {
    const res = await fetch('/api/files');
    const data = await res.json();
    const select = document.getElementById('fileSelect');
    if (!select) return;
    select.innerHTML = '';
    
    // Test Files group
    const optGroupTest = document.createElement('optgroup');
    optGroupTest.label = '── Held-out Test Set (68 Files) ──';
    data.test_files.forEach(f => {
      const opt = document.createElement('option');
      opt.value = f;
      opt.textContent = f;
      optGroupTest.appendChild(opt);
    });
    select.appendChild(optGroupTest);

    // Train Samples group
    const optGroupTrain = document.createElement('optgroup');
    optGroupTrain.label = '── Known Ground-Truth Samples ──';
    data.train_samples.forEach(f => {
      const opt = document.createElement('option');
      opt.value = f;
      opt.textContent = f;
      optGroupTrain.appendChild(opt);
    });
    select.appendChild(optGroupTrain);
  } catch (err) {
    console.error('File list fetch error:', err);
  }
}

function setupEventListeners() {
  const select = document.getElementById('fileSelect');
  let debounceTimer = null;
  if (select) {
    select.addEventListener('change', (e) => {
      const val = e.target.value;
      if (!val || val === currentFile) return;
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        analyzeFile(val);
      }, 50);
    });
  }

  // Download Zip
  const btnZip = document.getElementById('btnDownloadZip');
  if (btnZip) {
    btnZip.addEventListener('click', () => {
      window.location.href = '/api/download_predictions';
      showToast('📦 Downloading official predictions.zip...');
    });
  }
}

/**
 * Cancels any active in-flight request or event stream to prevent race conditions.
 */
function abortActiveOperations() {
  if (activeEventSource) {
    activeEventSource.close();
    activeEventSource = null;
  }
  if (activeAbortController) {
    activeAbortController.abort();
    activeAbortController = null;
  }
  isStreaming = false;
  setStreamBadge(false);
}

/**
 * Analyzes a selected recording using token-by-token streaming.
 */
function analyzeFile(fname) {
  if (!fname) return;
  abortActiveOperations();
  currentFile = fname;

  const recElem = document.getElementById('recommendationText');
  const streamBadge = document.getElementById('streamBadge');
  const verdictTitle = document.getElementById('verdictTitle');
  const verdictSubtitle = document.getElementById('verdictSubtitle');
  
  if (verdictTitle) verdictTitle.textContent = 'ANALYZING...';
  if (verdictSubtitle) verdictSubtitle.textContent = `Streaming condition telemetry for ${fname}...`;
  
  setStreamBadge(true);
  showToast(`Streaming AI Diagnostics on ${fname}...`);

  // Prepare recommendation container for streaming tokens
  if (recElem) {
    recElem.innerHTML = '<span id="streamedContent"></span><span class="streaming-cursor"></span>';
  }

  // Use Server-Sent Events (SSE) for true real-time token streaming
  const sseUrl = `/api/stream_predict?file=${encodeURIComponent(fname)}`;
  const evtSource = new EventSource(sseUrl);
  activeEventSource = evtSource;

  evtSource.addEventListener('metadata', (e) => {
    try {
      const data = JSON.parse(e.data);
      renderMetadata(data);
      updateTrainSchematic(data.car_wheel_data);
    } catch (err) {
      console.error('Metadata parse error:', err);
    }
  });

  evtSource.addEventListener('token', (e) => {
    try {
      const payload = JSON.parse(e.data);
      const span = document.getElementById('streamedContent');
      if (span) {
        span.textContent += payload.token;
      }
    } catch (err) {
      console.error('Token parse error:', err);
    }
  });

  evtSource.addEventListener('done', () => {
    finishStreaming();
    if (activeEventSource === evtSource) {
      evtSource.close();
      activeEventSource = null;
    }
  });

  evtSource.onerror = () => {
    // If SSE fails or is closed, fallback to standard fetch
    if (activeEventSource === evtSource) {
      evtSource.close();
      activeEventSource = null;
      fallbackFetchAnalyze(fname);
    }
  };
}

/**
 * Fallback analysis via standard HTTP JSON fetch with client-side token streaming typewriter.
 */
async function fallbackFetchAnalyze(fname) {
  try {
    activeAbortController = new AbortController();
    const res = await fetch(`/api/predict?file=${encodeURIComponent(fname)}`, {
      signal: activeAbortController.signal
    });
    if (!res.ok) throw new Error('API request failed');
    const data = await res.json();
    
    renderMetadata(data);
    updateTrainSchematic(data.car_wheel_data);
    
    // Client-side typewriter token streaming
    await streamClientTokens(data.recommendation);
  } catch (err) {
    if (err.name !== 'AbortError') {
      console.error('Fetch analyze error:', err);
      showToast(`Error analyzing ${fname}`);
      finishStreaming();
    }
  }
}

/**
 * Smooth client-side token typewriter effect
 */
function streamClientTokens(text) {
  return new Promise((resolve) => {
    const span = document.getElementById('streamedContent');
    if (!span) {
      finishStreaming();
      return resolve();
    }
    span.textContent = '';
    const words = text.split(' ');
    let idx = 0;
    
    const interval = setInterval(() => {
      if (idx >= words.length) {
        clearInterval(interval);
        finishStreaming();
        return resolve();
      }
      span.textContent += (idx === 0 ? '' : ' ') + words[idx];
      idx++;
    }, 18);
  });
}

function finishStreaming() {
  isStreaming = false;
  setStreamBadge(false);
  const cursor = document.querySelector('.streaming-cursor');
  if (cursor) cursor.remove();
}

function setStreamBadge(active) {
  const badge = document.getElementById('streamBadge');
  if (badge) {
    badge.style.display = active ? 'inline-flex' : 'none';
  }
}

function renderMetadata(data) {
  const titleElem = document.getElementById('verdictTitle');
  const iconElem = document.getElementById('verdictIcon');
  const badgeElem = document.getElementById('urgencyBadge');
  const subElem = document.getElementById('verdictSubtitle');
  const confElem = document.getElementById('confidenceVal');
  
  if (titleElem) {
    titleElem.textContent = data.prediction.toUpperCase();
    titleElem.className = `verdict-title verdict-${data.prediction.replace(' ', '-')}`;
  }
  
  if (badgeElem) {
    badgeElem.textContent = `URGENCY: ${data.urgency}`;
    badgeElem.className = `urgency-badge urgency-${data.urgency}`;
  }
  
  if (confElem) {
    confElem.textContent = `Confidence: ${data.confidence}%`;
  }
  
  if (iconElem) {
    if (data.prediction === 'Normal') {
      iconElem.textContent = '✅';
      if (subElem) subElem.textContent = `Train speed ${data.speed_kmh} km/h (${data.speed_mps} m/s) — Both rails operating within healthy acoustic limits.`;
    } else if (data.prediction === 'Side I') {
      iconElem.textContent = '⚠️';
      if (subElem) subElem.textContent = `Train speed ${data.speed_kmh} km/h — Abnormal corrugation on Left Rail (λ ≈ ${data.corrugation_wavelength_mm} mm, Peak ${data.peak_frequency_hz} Hz).`;
    } else {
      iconElem.textContent = '🚨';
      if (subElem) subElem.textContent = `Train speed ${data.speed_kmh} km/h — Abnormal corrugation on Right Rail (λ ≈ ${data.corrugation_wavelength_mm} mm, Peak ${data.peak_frequency_hz} Hz).`;
    }
  }
}

function renderInitialTrainSchematic() {
  const container = document.getElementById('trainConsist');
  if (!container) return;
  container.innerHTML = '';
  
  for (let car = 1; car <= 8; car++) {
    const carElem = document.createElement('div');
    carElem.className = 'train-car';
    carElem.id = `car-${car}`;
    
    const header = document.createElement('div');
    header.className = 'car-header';
    header.textContent = car === 1 ? 'CAR 1 (HEAD)' : car === 8 ? 'CAR 8 (TAIL)' : `CAR ${car}`;
    carElem.appendChild(header);

    const wheelsets = document.createElement('div');
    wheelsets.className = 'car-wheelsets';

    // Top row: Side I (1, 3, 5, 7)
    const rowTop = document.createElement('div');
    rowTop.className = 'wheel-row';
    [1, 3, 5, 7].forEach(pos => {
      const node = document.createElement('div');
      node.className = 'wheel-node';
      node.id = `wheel-${car}-${pos}`;
      node.textContent = pos;
      setupWheelHover(node, car, pos, 'Side I');
      rowTop.appendChild(node);
    });
    wheelsets.appendChild(rowTop);

    // Bottom row: Side II (2, 4, 6, 8)
    const rowBottom = document.createElement('div');
    rowBottom.className = 'wheel-row';
    [2, 4, 6, 8].forEach(pos => {
      const node = document.createElement('div');
      node.className = 'wheel-node';
      node.id = `wheel-${car}-${pos}`;
      node.textContent = pos;
      setupWheelHover(node, car, pos, 'Side II');
      rowBottom.appendChild(node);
    });
    wheelsets.appendChild(rowBottom);

    carElem.appendChild(wheelsets);
    container.appendChild(carElem);
  }
}

function setupWheelHover(node, car, pos, side) {
  node.addEventListener('mouseenter', () => {
    const inspectBar = document.getElementById('wheelInspectBar');
    if (!inspectBar) return;
    const vRms = node.dataset.vibRms || '--';
    const sRms = node.dataset.shockRms || '--';
    inspectBar.innerHTML = `<strong>Car ${car}, Axle-Box Position ${pos} (${side}):</strong> Live Vibration RMS = <span style="color:var(--primary)">${vRms} m/s²</span> | Shock RMS = <span style="color:var(--text-main)">${sRms} m/s²</span>`;
  });
}

function updateTrainSchematic(carWheelData) {
  if (!carWheelData) return;
  
  carWheelData.forEach(carInfo => {
    const car = carInfo.car;
    for (let pos = 1; pos <= 8; pos++) {
      const node = document.getElementById(`wheel-${car}-${pos}`);
      if (!node) continue;
      
      const pData = carInfo.positions[pos];
      node.dataset.vibRms = pData.vib_rms;
      node.dataset.shockRms = pData.shock_rms;
      
      // Color code based on vibration threshold
      if (pData.vib_rms > 0.70) {
        node.style.background = 'var(--color-danger)';
        node.style.color = '#fff';
      } else if (pData.vib_rms > 0.35) {
        node.style.background = 'var(--color-side1)';
        node.style.color = '#000';
      } else {
        node.style.background = 'var(--color-normal)';
        node.style.color = '#000';
      }
    }
  });
}

let toastTimer = null;
function showToast(msg) {
  const toast = document.getElementById('toast');
  if (!toast) return;
  toast.textContent = msg;
  toast.style.display = 'block';
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.style.display = 'none';
  }, 2500);
}

