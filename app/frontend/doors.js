// NEBULA-X: Train Doors Subsystem Client Script

let doorsData = null;
let currentFilter = 'all';
let isObstacleActive = false;
let isDoorOpen = false;

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  fetchDoorsPredictions();
  initDoorSimulator();
  handleHashNavigation();
});

// 1. Tab Switching & Deep Links
function initTabs() {
  const tabBtns = document.querySelectorAll('.sub-tab-btn');
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
    drawDoorWaveform(isObstacleActive);
  }
}

function handleHashNavigation() {
  const hash = window.location.hash;
  if (hash === '#analysis') switchTab('pane-analysis');
  else if (hash === '#twin') switchTab('pane-twin');
  else switchTab('pane-predictions');
}

// 2. Fetch & Render Door Predictions
async function fetchDoorsPredictions() {
  try {
    const res = await fetch('/api/doors/predictions');
    const data = await res.json();
    doorsData = data;

    if (data.summary) {
      document.getElementById('doorKpiTotal').textContent = data.summary.total_cycles;
      document.getElementById('doorKpiNormal').textContent = data.summary.normal_cycles;
      document.getElementById('doorKpiAbnormal').textContent = data.summary.abnormal_resistance;
    }

    renderTable();
    initFilters();
  } catch (err) {
    console.error('Failed to fetch doors predictions:', err);
  }
}

function initFilters() {
  const filterBtns = document.querySelectorAll('.filter-btn');
  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentFilter = btn.getAttribute('data-filter');
      renderTable();
    });
  });

  const searchInput = document.getElementById('doorSearchInput');
  if (searchInput) {
    searchInput.addEventListener('input', () => {
      renderTable();
    });
  }
}

function renderTable() {
  const tbody = document.getElementById('doorsTableBody');
  if (!tbody || !doorsData || !doorsData.cycles) return;

  const searchQuery = (document.getElementById('doorSearchInput')?.value || '').trim().toLowerCase();
  tbody.innerHTML = '';

  const filtered = doorsData.cycles.filter(c => {
    if (currentFilter === 'normal' && c.label !== 'Normal') return false;
    if (currentFilter === 'abnormal' && c.label !== 'Abnormal Resistance') return false;
    if (searchQuery && !c.cycle_id.toString().includes(searchQuery)) return false;
    return true;
  });

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-dim); padding: 2rem;">No matching door cycles found.</td></tr>`;
    return;
  }

  filtered.forEach(row => {
    const tr = document.createElement('tr');
    const isAbnormal = row.label === 'Abnormal Resistance';

    const badgeColor = isAbnormal ? 'var(--color-danger)' : 'var(--color-normal)';
    const badgeBg = isAbnormal ? 'rgba(239, 68, 68, 0.15)' : 'rgba(16, 185, 129, 0.15)';
    const badgeBorder = isAbnormal ? 'rgba(239, 68, 68, 0.3)' : 'rgba(16, 185, 129, 0.3)';

    tr.innerHTML = `
      <td style="font-family: var(--font-mono); font-weight: 700; color: #fff;">#${row.cycle_id}</td>
      <td style="font-size: 0.85rem;">${row.direction}</td>
      <td style="font-family: var(--font-mono);">${row.duration_seconds} s</td>
      <td style="font-family: var(--font-mono); font-weight: 700; color: ${isAbnormal ? '#ef4444' : '#fff'};">${row.peak_current_a} A</td>
      <td>
        <span style="display: inline-block; padding: 0.2rem 0.65rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; color: ${badgeColor}; background: ${badgeBg}; border: 1px solid ${badgeBorder};">
          ${row.label}
        </span>
      </td>
      <td style="font-family: var(--font-mono); font-weight: 600; color: ${badgeColor};">${row.anomaly_score.toFixed(3)}</td>
      <td style="font-size: 0.8rem; color: var(--text-muted);">${row.action}</td>
      <td>
        <button class="btn btn-primary" style="padding: 0.3rem 0.65rem; font-size: 0.75rem;" onclick="simulateDoorCycle(${row.cycle_id}, ${isAbnormal})">
          Simulate 🔬
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

window.simulateDoorCycle = function(cycleId, isAbnormal) {
  switchTab('pane-twin');
  isObstacleActive = isAbnormal;
  
  const obstacleEl = document.getElementById('doorObstacleVisualizer');
  const toggleBtn = document.getElementById('btnToggleObstacle');
  if (obstacleEl) {
    if (isAbnormal) obstacleEl.classList.add('active');
    else obstacleEl.classList.remove('active');
  }
  if (toggleBtn) {
    toggleBtn.textContent = isAbnormal ? '✓ Obstacle Active (Click to Remove)' : '⚠️ Inject Guide Rail Obstacle';
  }

  triggerDoorCycle(true);
};

// 3. Interactive Door Twin Simulator & Waveform
function initDoorSimulator() {
  const btnOpen = document.getElementById('btnOpenDoors');
  const btnClose = document.getElementById('btnCloseDoors');
  const btnToggleObs = document.getElementById('btnToggleObstacle');

  if (btnOpen) btnOpen.addEventListener('click', () => triggerDoorCycle(true));
  if (btnClose) btnClose.addEventListener('click', () => triggerDoorCycle(false));

  if (btnToggleObs) {
    btnToggleObs.addEventListener('click', () => {
      isObstacleActive = !isObstacleActive;
      const obsEl = document.getElementById('doorObstacleVisualizer');
      if (obsEl) {
        if (isObstacleActive) obsEl.classList.add('active');
        else obsEl.classList.remove('active');
      }
      btnToggleObs.textContent = isObstacleActive ? '✓ Obstacle Active (Click to Remove)' : '⚠️ Inject Guide Rail Obstacle';
      drawDoorWaveform(isObstacleActive);
    });
  }

  drawDoorWaveform(false);
}

function triggerDoorCycle(open) {
  isDoorOpen = open;
  const frame = document.getElementById('doorFrameVisualizer');
  const label = document.getElementById('doorStateLabel');

  if (open) {
    frame?.classList.add('open');
    if (label) {
      label.textContent = 'STATUS: OPENING MOTION';
      label.style.color = '#00f0ff';
    }
  } else {
    frame?.classList.remove('open');
    if (label) {
      label.textContent = 'STATUS: CLOSING MOTION';
      label.style.color = '#10b981';
    }
  }

  // Update simulator stats
  const peakCur = isObstacleActive ? (7.4 + Math.random() * 0.8).toFixed(2) : (3.2 + Math.random() * 0.4).toFixed(2);
  const strokeDur = isObstacleActive ? (4.15 + Math.random() * 0.3).toFixed(2) : (2.80 + Math.random() * 0.1).toFixed(2);

  document.getElementById('simPeakCurrent').textContent = `${peakCur} A`;
  document.getElementById('simStrokeDuration').textContent = `${strokeDur} s`;

  const dragStatus = document.getElementById('simDragStatus');
  const badge = document.getElementById('doorSimStatusBadge');

  if (isObstacleActive) {
    if (dragStatus) {
      dragStatus.textContent = 'OBSTACLE SPIKE';
      dragStatus.style.color = '#ef4444';
    }
    if (badge) {
      badge.textContent = '🚨 ABNORMAL RESISTANCE';
      badge.style.color = '#ef4444';
      badge.style.background = 'rgba(239, 68, 68, 0.2)';
      badge.style.borderColor = 'rgba(239, 68, 68, 0.4)';
    }
  } else {
    if (dragStatus) {
      dragStatus.textContent = 'NOMINAL';
      dragStatus.style.color = 'var(--color-normal)';
    }
    if (badge) {
      badge.textContent = '● NORMAL MOTION';
      badge.style.color = 'var(--color-normal)';
      badge.style.background = 'rgba(16, 185, 129, 0.15)';
      badge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
    }
  }

  drawDoorWaveform(isObstacleActive);

  setTimeout(() => {
    if (label) {
      label.textContent = open ? 'STATUS: FULLY OPEN' : 'STATUS: FULLY CLOSED';
      label.style.color = isObstacleActive ? '#ef4444' : 'var(--color-normal)';
    }
  }, 1300);
}

function drawDoorWaveform(abnormal = false) {
  const canvas = document.getElementById('doorWaveformCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  const padLeft = 60;
  const padBottom = 35;
  const padTop = 20;
  const padRight = 20;

  // Grid
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
  ctx.lineWidth = 1;
  for (let x = padLeft; x < w - padRight; x += 50) {
    ctx.beginPath();
    ctx.moveTo(x, padTop);
    ctx.lineTo(x, h - padBottom);
    ctx.stroke();
  }
  for (let y = padTop; y < h - padBottom; y += 35) {
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(w - padRight, y);
    ctx.stroke();
  }

  // Axes
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(padLeft, padTop);
  ctx.lineTo(padLeft, h - padBottom);
  ctx.lineTo(w - padRight, h - padBottom);
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = '#94a3b8';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillText('0s', padLeft, h - padBottom + 18);
  ctx.fillText('1s', padLeft + 120, h - padBottom + 18);
  ctx.fillText('2s', padLeft + 240, h - padBottom + 18);
  ctx.fillText('3s', padLeft + 360, h - padBottom + 18);
  ctx.fillText('4s', padLeft + 480, h - padBottom + 18);
  ctx.fillText('5s', padLeft + 600, h - padBottom + 18);

  ctx.fillText('8A', 20, padTop + 10);
  ctx.fillText('6A', 20, padTop + 55);
  ctx.fillText('4A', 20, padTop + 105);
  ctx.fillText('2A', 20, padTop + 155);
  ctx.fillText('0A', 20, h - padBottom);

  // Generate 50Hz samples
  const plotW = w - padLeft - padRight;
  const plotH = h - padTop - padBottom;

  ctx.strokeStyle = abnormal ? '#ef4444' : '#10b981';
  ctx.lineWidth = 2.5;
  ctx.beginPath();

  const numPoints = 200;
  for (let i = 0; i <= numPoints; i++) {
    const t = (i / numPoints) * 4.5; // 0 to 4.5 seconds
    let current = 0;

    if (t < 0.2) {
      current = 0.0;
    } else if (t < 0.6) {
      // Inrush peak
      current = 1.0 + (t - 0.2) * 6.5;
    } else if (t < 1.0) {
      // Settle
      current = 3.6 - (t - 0.6) * 5.0;
    } else if (t < 2.5) {
      // Cruise
      current = 1.3 + Math.sin(t * 15) * 0.15;
      if (abnormal && t > 1.4 && t < 2.3) {
        // Obstacle surge!
        current = 6.8 + Math.sin(t * 30) * 0.6;
      }
    } else if (t < 3.0) {
      // Deceleration
      if (abnormal) {
        current = 3.5 - (t - 2.5) * 4.0;
      } else {
        current = 1.3 - (t - 2.5) * 2.5;
      }
    } else {
      current = 0.05;
    }

    current = Math.max(0, current);
    const px = padLeft + (i / numPoints) * plotW;
    const py = padTop + plotH - (current / 8.0) * plotH;

    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();

  // Highlight threshold line at 6.0A
  ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  const threshY = padTop + plotH - (6.0 / 8.0) * plotH;
  ctx.moveTo(padLeft, threshY);
  ctx.lineTo(w - padRight, threshY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = '#ef4444';
  ctx.font = '10px Inter, sans-serif';
  ctx.fillText('Critical Resistance Threshold (6.0A)', w - padRight - 190, threshY - 5);
}
window.addEventListener('resize', () => drawDoorWaveform(isObstacleActive));
