// NEBULA-X: Structural Health Monitoring (SHM) Client Script

let shmPredictionsData = null;
let currentFilter = 'all';

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  fetchShmPredictions();
  initSimulator();
  handleHashNavigation();
});

// 1. Navigation & Tab Switching
function initTabs() {
  const tabBtns = document.querySelectorAll('.sub-tab-btn');
  const navTabs = document.querySelectorAll('.nav-tabs .nav-tab');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-target');
      switchTab(targetId);
    });
  });

  document.getElementById('navTabPredictions')?.addEventListener('click', (e) => {
    e.preventDefault();
    switchTab('pane-predictions');
  });
  document.getElementById('navTabAnalysis')?.addEventListener('click', (e) => {
    e.preventDefault();
    switchTab('pane-analysis');
  });
  document.getElementById('navTabTwin')?.addEventListener('click', (e) => {
    e.preventDefault();
    switchTab('pane-twin');
  });
}

function switchTab(paneId) {
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.sub-tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.nav-tabs .nav-tab').forEach(t => t.classList.remove('active'));

  const targetPane = document.getElementById(paneId);
  if (targetPane) targetPane.classList.add('active');

  const activeBtn = document.querySelector(`.sub-tab-btn[data-target="${paneId}"]`);
  if (activeBtn) activeBtn.classList.add('active');

  if (paneId === 'pane-predictions') {
    document.getElementById('navTabPredictions')?.classList.add('active');
    window.location.hash = 'predictions';
  } else if (paneId === 'pane-analysis') {
    document.getElementById('navTabAnalysis')?.classList.add('active');
    window.location.hash = 'analysis';
  } else if (paneId === 'pane-twin') {
    document.getElementById('navTabTwin')?.classList.add('active');
    window.location.hash = 'twin';
    drawSnCurve();
  }
}

function handleHashNavigation() {
  const hash = window.location.hash;
  if (hash === '#analysis') {
    switchTab('pane-analysis');
  } else if (hash === '#twin') {
    switchTab('pane-twin');
  } else {
    switchTab('pane-predictions');
  }
}

// 2. Fetch & Render SHM Predictions
async function fetchShmPredictions() {
  try {
    const res = await fetch('/api/shm/predictions');
    const data = await res.json();
    shmPredictionsData = data;

    // Populate KPIs & Tab Title
    if (data.summary) {
      document.getElementById('shmKpiTotal').textContent = data.summary.total_files;
      document.getElementById('shmKpiLow').textContent = data.summary.low_risk;
      document.getElementById('shmKpiMed').textContent = data.summary.medium_risk;
      document.getElementById('shmKpiHigh').textContent = data.summary.high_risk;
      document.getElementById('shmKpiAvgD').textContent = data.summary.avg_damage_index.toFixed(4);
      
      const tabTitle = document.getElementById('shmTabTitle');
      if (tabTitle) tabTitle.textContent = `Predictions Overview (${data.summary.total_files} Dynamic Stress Cases)`;

      const cAll = document.getElementById('countAllShm');
      const cLow = document.getElementById('countLowShm');
      const cMed = document.getElementById('countMedShm');
      const cHigh = document.getElementById('countHighShm');
      if (cAll) cAll.textContent = data.summary.total_files;
      if (cLow) cLow.textContent = data.summary.low_risk;
      if (cMed) cMed.textContent = data.summary.medium_risk;
      if (cHigh) cHigh.textContent = data.summary.high_risk;

      const uploadedCount = (data.predictions || []).filter(p => p.is_uploaded).length;
      const btnUp = document.getElementById('btnFilterUploadedShm');
      const countUp = document.getElementById('countUploadedShm');
      if (btnUp && countUp) {
        countUp.textContent = uploadedCount;
        btnUp.style.display = uploadedCount > 0 ? 'inline-block' : 'none';
      }
    }

    renderTable();
    initFilters();
    initShmUpload();
  } catch (err) {
    console.error('Failed to fetch SHM predictions:', err);
  }
}

function initShmUpload() {
  const btnUpload = document.getElementById('btnUploadShm');
  const fileInput = document.getElementById('shmFileInput');
  if (!btnUpload || !fileInput || btnUpload._bound) return;
  btnUpload._bound = true;

  btnUpload.addEventListener('click', () => {
    fileInput.click();
  });

  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.csv')) {
      alert('Please select a valid single-column stress CSV file.');
      return;
    }

    const origText = btnUpload.innerHTML;
    btnUpload.disabled = true;
    btnUpload.innerHTML = '<span class="btn-icon">⏳</span> Computing Rainflow...';

    try {
      const res = await fetch('/api/shm/upload', {
        method: 'POST',
        headers: {
          'Content-Type': 'text/csv',
          'X-Filename': file.name
        },
        body: file
      });

      const data = await res.json();
      if (!res.ok || !data.success) {
        alert(`❌ SHM Analysis Failed:\n\n${data.error || 'Unknown error'}`);
        return;
      }

      alert(`✅ Stress Telemetry Verified & Diagnosed!\n\nFile: ${data.prediction.file_id}\nDamage Index (D): ${data.prediction.damage_index}\nSeverity: ${data.prediction.severity}\nRemaining Life: ${Number(data.prediction.remaining_hours).toLocaleString()} hours`);
      await fetchShmPredictions();
    } catch (err) {
      console.error('Upload failed:', err);
      alert(`Upload request failed: ${err.message}`);
    } finally {
      btnUpload.disabled = false;
      btnUpload.innerHTML = origText;
      fileInput.value = '';
    }
  });
}

function initFilters() {
  const filterBtns = document.querySelectorAll('.filter-btn');
  filterBtns.forEach(btn => {
    if (btn._bound) return;
    btn._bound = true;
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentFilter = btn.getAttribute('data-filter');
      renderTable();
    });
  });

  const searchInput = document.getElementById('shmSearchInput');
  if (searchInput && !searchInput._bound) {
    searchInput._bound = true;
    searchInput.addEventListener('input', () => renderTable());
  }
}

// Map API severity string to our internal filter key
function getSeverityFilter(severity) {
  if (!severity) return 'low';
  const s = severity.toLowerCase();
  if (s.startsWith('high')) return 'high';
  if (s.startsWith('medium')) return 'medium';
  return 'low';
}

// Map API severity string to display label & colours
function getSeverityStyle(severity) {
  const key = getSeverityFilter(severity);
  if (key === 'high') return { label: 'High Risk', color: 'var(--color-danger)', bg: 'rgba(239,68,68,0.15)', border: 'rgba(239,68,68,0.3)' };
  if (key === 'medium') return { label: 'Medium Risk', color: 'var(--color-side1)', bg: 'rgba(245,158,11,0.15)', border: 'rgba(245,158,11,0.3)' };
  return { label: 'Low Risk', color: 'var(--color-normal)', bg: 'rgba(16,185,129,0.15)', border: 'rgba(16,185,129,0.3)' };
}

function renderTable() {
  const tbody = document.getElementById('shmTableBody');
  if (!tbody || !shmPredictionsData || !shmPredictionsData.predictions) return;

  const searchQuery = (document.getElementById('shmSearchInput')?.value || '').toLowerCase();
  tbody.innerHTML = '';

  const filtered = shmPredictionsData.predictions.filter(item => {
    if (currentFilter === 'uploaded' && !item.is_uploaded) return false;
    if (currentFilter !== 'all' && currentFilter !== 'uploaded' && getSeverityFilter(item.severity) !== currentFilter) return false;
    if (searchQuery && !item.file_id.toLowerCase().includes(searchQuery)) return false;
    return true;
  });

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-dim); padding: 2rem;">No matching stress recordings found.</td></tr>`;
    return;
  }

  filtered.forEach(row => {
    const tr = document.createElement('tr');

    // Map API fields to display values
    const damageIndex   = row.damage_index   ?? row.prediction ?? 0;
    const remainingHrs  = row.est_remaining_hours ?? row.remaining_hours ?? 0;
    const peakStress    = row.peak_stress_range_mpa ?? row.stress_range ?? row.max_stress ?? 0;
    const cycleCount    = row.cycle_count    ?? row.cycles   ?? 0;
    const action        = row.recommended_action ?? row.notes ?? '—';

    const style = getSeverityStyle(row.severity);
    const dPct  = Math.min(100, Math.round(damageIndex * 100));

    tr.innerHTML = `
      <td style="font-family: var(--font-mono); font-weight: 600; color: #fff;">${row.file_id}</td>
      <td>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
          <div style="flex-grow: 1; height: 6px; background: rgba(255,255,255,0.08); border-radius: 3px; overflow: hidden; width: 70px;">
            <div style="width: ${dPct}%; height: 100%; background: ${style.color}; border-radius: 3px;"></div>
          </div>
          <span style="font-family: var(--font-mono); font-weight: 700; color: ${style.color};">${damageIndex.toFixed(4)}</span>
        </div>
      </td>
      <td>
        <span style="display: inline-block; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; color: ${style.color}; background: ${style.bg}; border: 1px solid ${style.border};">
          ${style.label}
        </span>
      </td>
      <td style="font-family: var(--font-mono);">${Number(remainingHrs).toLocaleString()} hrs</td>
      <td style="font-family: var(--font-mono);">${Number(peakStress).toFixed(1)} MPa</td>
      <td style="font-family: var(--font-mono);">${Number(cycleCount).toLocaleString()}</td>
      <td style="font-size: 0.8rem; color: var(--text-muted);">${action}</td>
      <td>
        <button class="btn btn-primary" style="padding: 0.3rem 0.65rem; font-size: 0.75rem;"
          onclick="inspectInTwin('${row.file_id}', ${damageIndex}, ${Number(peakStress).toFixed(1)}, ${cycleCount})">
          Inspect 🔬
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

// 3. Inspect in Digital Twin Action
window.inspectInTwin = function(fileId, dVal, stressVal, cycles) {
  switchTab('pane-twin');
  document.getElementById('twinActiveFileBadge').textContent = `Active: ${fileId}`;
  
  const sliderStress = document.getElementById('sliderStress');
  const sliderCycles = document.getElementById('sliderCycles');
  if (sliderStress) sliderStress.value = stressVal;
  if (sliderCycles) sliderCycles.value = Math.min(250000, cycles);
  
  updateSimulator();
};

// 4. Interactive Fatigue Twin & S-N Wöhler Curve Simulator
function initSimulator() {
  const sliderStress = document.getElementById('sliderStress');
  const sliderCycles = document.getElementById('sliderCycles');
  const sliderLoad = document.getElementById('sliderLoad');

  if (sliderStress) sliderStress.addEventListener('input', updateSimulator);
  if (sliderCycles) sliderCycles.addEventListener('input', updateSimulator);
  if (sliderLoad) sliderLoad.addEventListener('input', updateSimulator);

  updateSimulator();
}

function updateSimulator() {
  const stress = parseFloat(document.getElementById('sliderStress')?.value || 120);
  const cycles = parseFloat(document.getElementById('sliderCycles')?.value || 50000);
  const load = parseFloat(document.getElementById('sliderLoad')?.value || 1.2);

  // Update labels
  document.getElementById('sliderStressVal').textContent = `${stress} MPa`;
  document.getElementById('sliderCyclesVal').textContent = `${cycles.toLocaleString()} cycles`;
  document.getElementById('sliderLoadVal').textContent = `${load.toFixed(1)}x`;

  // Basquin's Relation: N_fail = C / (Δσ * load)^m
  // For structural bogie steel (e.g. S355): m = 3.5, C = 2.0e12
  const effectiveStress = stress * load;
  const N_fail = 2.0e12 / Math.pow(effectiveStress, 3.5);
  const D = Math.min(1.0, Math.max(0.0001, cycles / N_fail));

  // Remaining useful hours (assuming 2500 operating hours / year)
  const remainingFraction = Math.max(0, 1.0 - D);
  const remainingHours = Math.round(remainingFraction * 9000);
  const remainingYears = (remainingHours / 2000).toFixed(1);

  document.getElementById('simulatedDVal').textContent = D.toFixed(4);
  document.getElementById('simulatedRulVal').textContent = `${remainingHours.toLocaleString()} hrs (~${remainingYears} yrs)`;

  const badge = document.getElementById('simulatedStatusBadge');
  if (badge) {
    if (D < 0.10) {
      badge.style.color = 'var(--color-normal)';
      badge.style.background = 'rgba(16, 185, 129, 0.2)';
      badge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
      badge.textContent = '● Low Risk • Full Structural Margin';
    } else if (D < 0.50) {
      badge.style.color = 'var(--color-side1)';
      badge.style.background = 'rgba(245, 158, 11, 0.2)';
      badge.style.borderColor = 'rgba(245, 158, 11, 0.4)';
      badge.textContent = '● Medium Risk • Scheduled Ultrasonic NDT';
    } else {
      badge.style.color = 'var(--color-danger)';
      badge.style.background = 'rgba(239, 68, 68, 0.2)';
      badge.style.borderColor = 'rgba(239, 68, 68, 0.4)';
      badge.textContent = '● High Risk • Critical Fatigue Warning!';
    }
  }

  drawSnCurve(effectiveStress, cycles);
}

function drawSnCurve(currentStress = 144, currentCycles = 50000) {
  const canvas = document.getElementById('snCurveCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Background grid
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
  ctx.lineWidth = 1;
  const padLeft = 70;
  const padBottom = 40;
  const padTop = 20;
  const padRight = 30;

  for (let x = padLeft; x < w - padRight; x += 60) {
    ctx.beginPath();
    ctx.moveTo(x, padTop);
    ctx.lineTo(x, h - padBottom);
    ctx.stroke();
  }

  for (let y = padTop; y < h - padBottom; y += 40) {
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(w - padRight, y);
    ctx.stroke();
  }

  // Draw Axes
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(padLeft, padTop);
  ctx.lineTo(padLeft, h - padBottom);
  ctx.lineTo(w - padRight, h - padBottom);
  ctx.stroke();

  // Axis Labels
  ctx.fillStyle = '#94a3b8';
  ctx.font = '11px JetBrains Mono, monospace';
  ctx.fillText('10³', padLeft + 20, h - padBottom + 20);
  ctx.fillText('10⁴', padLeft + 160, h - padBottom + 20);
  ctx.fillText('10⁵', padLeft + 320, h - padBottom + 20);
  ctx.fillText('10⁶', padLeft + 480, h - padBottom + 20);
  ctx.fillText('10⁷', padLeft + 640, h - padBottom + 20);

  ctx.fillText('300 MPa', 10, padTop + 10);
  ctx.fillText('200 MPa', 10, padTop + 90);
  ctx.fillText('100 MPa', 10, padTop + 170);
  ctx.fillText('30 MPa', 10, h - padBottom - 10);

  // Draw Basquin S-N Curve: log(N) vs Stress
  ctx.strokeStyle = '#f59e0b';
  ctx.lineWidth = 3;
  ctx.beginPath();

  const plotW = w - padLeft - padRight;
  const plotH = h - padTop - padBottom;

  for (let px = 0; px <= plotW; px += 4) {
    const logN = 3 + (px / plotW) * 4; // 10^3 to 10^7
    const N = Math.pow(10, logN);
    // Stress = (C / N)^(1/m)
    const curveStress = Math.pow(2.0e12 / N, 1 / 3.5);
    const py = padTop + plotH - ((curveStress - 30) / (300 - 30)) * plotH;

    if (px === 0) ctx.moveTo(padLeft + px, Math.max(padTop, Math.min(h - padBottom, py)));
    else ctx.lineTo(padLeft + px, Math.max(padTop, Math.min(h - padBottom, py)));
  }
  ctx.stroke();

  // Draw Current Operating Point
  const currentLogN = Math.log10(Math.max(1000, currentCycles));
  const ptX = padLeft + ((currentLogN - 3) / 4) * plotW;
  const ptY = padTop + plotH - ((currentStress - 30) / (300 - 30)) * plotH;

  if (ptX >= padLeft && ptX <= w - padRight && ptY >= padTop && ptY <= h - padBottom) {
    // Glowing circle
    ctx.fillStyle = '#00f0ff';
    ctx.shadowColor = '#00f0ff';
    ctx.shadowBlur = 15;
    ctx.beginPath();
    ctx.arc(ptX, ptY, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;

    // Label
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.fillText(`Δσ: ${currentStress.toFixed(0)} MPa`, ptX + 10, ptY - 8);
  }
}
window.addEventListener('resize', () => drawSnCurve());
