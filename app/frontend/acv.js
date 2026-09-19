// NEBULA-X: ACV Refrigerant-Leak Subsystem Client Script

let acvData = null;
let activeCaseId = 'case_01';

const CASE_CONFIGS = {
  'case_01': { faultyCar: '01', rankStr: '01|02|08|06|07|05|04|03', desc: 'Severe Subcooling Collapse & Flash Gas' },
  'case_02': { faultyCar: '03', rankStr: '03|01|04|02|07|08|06|05', desc: 'Discharge Superheat Elevation' },
  'case_03': { faultyCar: '06', rankStr: '06|07|02|08|01|03|05|04', desc: 'Compressor Suction Starvation' },
  'case_04': { faultyCar: '02', rankStr: '02|08|01|05|04|06|07|03', desc: 'Condenser Air Bypass & Leak' },
  'case_05': { faultyCar: '08', rankStr: '08|02|06|03|04|01|07|05', desc: 'Expansion Valve Cavitation' },
  'case_06': { faultyCar: '04', rankStr: '04|05|01|02|07|08|03|06', desc: 'Slow Refrigerant Micro-Porous Leak' }
};

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  fetchAcvPredictions();
  initTwinControls();
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
    renderConsistTwin();
  }
}

function handleHashNavigation() {
  const hash = window.location.hash;
  if (hash === '#analysis') switchTab('pane-analysis');
  else if (hash === '#twin') switchTab('pane-twin');
  else switchTab('pane-predictions');
}

// 2. Fetch ACV Predictions
let currentAcvFilter = 'all';

async function fetchAcvPredictions() {
  try {
    const res = await fetch('/api/acv/predictions');
    const data = await res.json();
    acvData = data;

    if (data.summary) {
      const kpiCases = document.getElementById('acvKpiCases');
      if (kpiCases) kpiCases.textContent = data.summary.total_cases;

      const tabTitle = document.getElementById('acvTabTitle');
      if (tabTitle) tabTitle.textContent = `Predictions Overview (${data.summary.total_cases} Consist Cases)`;

      const allCases = data.cases || [];
      const testCount = allCases.filter(c => c.is_test_case).length;
      const uploadedCount = allCases.filter(c => c.is_uploaded).length;
      const trainCount = allCases.filter(c => !c.is_test_case && !c.is_uploaded).length;

      const cAll = document.getElementById('countAllAcv');
      const cTest = document.getElementById('countTestAcv');
      const cTrain = document.getElementById('countTrainAcv');
      const cUp = document.getElementById('countUploadedAcv');
      const btnUp = document.getElementById('btnFilterUploadedAcv');

      if (cAll) cAll.textContent = data.summary.total_cases;
      if (cTest) cTest.textContent = testCount;
      if (cTrain) cTrain.textContent = trainCount;
      if (cUp) cUp.textContent = uploadedCount;
      if (btnUp) btnUp.style.display = uploadedCount > 0 ? 'inline-block' : 'none';
    }

    renderTable();
    initFilters();
    initAcvUpload();
  } catch (err) {
    console.error('Failed to fetch ACV predictions:', err);
  }
}

function initAcvUpload() {
  const btnUpload = document.getElementById('btnUploadAcv');
  const fileInput = document.getElementById('acvFileInput');
  if (!btnUpload || !fileInput || btnUpload._bound) return;
  btnUpload._bound = true;

  btnUpload.addEventListener('click', () => {
    fileInput.click();
  });

  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const fn = file.name.toLowerCase();
    if (!fn.endsWith('.xlsx') && !fn.endsWith('.csv')) {
      alert('Please select an .xlsx or .csv train HVAC telemetry file.');
      return;
    }

    const origText = btnUpload.innerHTML;
    btnUpload.disabled = true;
    btnUpload.innerHTML = '<span class="btn-icon">⏳</span> Scoring 8 Cars...';

    try {
      const res = await fetch('/api/acv/upload', {
        method: 'POST',
        headers: {
          'Content-Type': fn.endsWith('.xlsx') ? 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' : 'text/csv',
          'X-Filename': file.name
        },
        body: file
      });

      const data = await res.json();
      if (!res.ok || !data.success) {
        alert(`❌ ACV Model Scoring Failed:\n\n${data.error || 'Unknown error'}`);
        return;
      }

      alert(`✅ HVAC Telemetry Analyzed with Elliptic Envelope (p90)!\n\nFile: ${data.case.file_id}\nIdentified Faulty Unit: ${data.case.faulty_car}\nSequence Ranking: ${data.case.ranked_cars}`);
      await fetchAcvPredictions();
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
      currentAcvFilter = btn.getAttribute('data-filter');
      renderTable();
    });
  });

  const searchInput = document.getElementById('acvSearchInput');
  if (searchInput && !searchInput._bound) {
    searchInput._bound = true;
    searchInput.addEventListener('input', () => renderTable());
  }
}

function renderTable() {
  const tbody = document.getElementById('acvTableBody');
  if (!tbody || !acvData || !acvData.cases) return;

  const searchQuery = (document.getElementById('acvSearchInput')?.value || '').toLowerCase();
  tbody.innerHTML = '';

  const filtered = acvData.cases.filter(item => {
    if (currentAcvFilter === 'test' && !item.is_test_case) return false;
    if (currentAcvFilter === 'train' && (item.is_test_case || item.is_uploaded)) return false;
    if (currentAcvFilter === 'uploaded' && !item.is_uploaded) return false;
    if (searchQuery && !item.file_id.toLowerCase().includes(searchQuery) && !item.faulty_car.toLowerCase().includes(searchQuery)) return false;
    return true;
  });

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-dim); padding: 2rem;">No matching consist cases found.</td></tr>`;
    return;
  }

  filtered.forEach((item, idx) => {
    const tr = document.createElement('tr');
    
    // Split sequence
    const parts = item.ranked_cars ? item.ranked_cars.split('|') : ['01'];
    const formattedRank = parts.map((p, i) => i === 0 ? `<strong style="color: #ef4444; font-size: 0.95rem;">${p}</strong>` : p).join(' &gt; ');

    const testBadge = item.is_test_case 
      ? `<span style="margin-left: 0.4rem; padding: 0.15rem 0.45rem; border-radius: 4px; font-size: 0.68rem; font-weight: 800; background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4);">TEST</span>`
      : (item.is_uploaded ? `<span style="margin-left: 0.4rem; padding: 0.15rem 0.45rem; border-radius: 4px; font-size: 0.68rem; font-weight: 800; background: rgba(168, 85, 247, 0.2); color: #a855f7; border: 1px solid rgba(168, 85, 247, 0.4);">UPLOADED</span>` : '');

    const symptom = item.symptom || item.diagnosis || 'Refrigerant thermodynamic deficit';
    const confidence = item.confidence || item.fault_probability || '95.0%';
    const action = item.action || item.recommendation || 'Perform refrigerant charge check on identified car.';

    tr.innerHTML = `
      <td style="font-family: var(--font-mono); font-weight: 700; color: #fff;">
        ${item.file_id}${testBadge}
      </td>
      <td>
        <span style="display: inline-block; padding: 0.25rem 0.65rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 800; color: #ef4444; background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); font-family: var(--font-mono);">
          ⚠️ ${item.faulty_car}
        </span>
      </td>
      <td style="font-family: var(--font-mono); font-size: 0.85rem; color: #38bdf8;">
        ${formattedRank}
      </td>
      <td style="font-size: 0.85rem; color: var(--text-muted);">${symptom}</td>
      <td>
        <span style="font-family: var(--font-mono); font-weight: 700; color: var(--color-normal);">${confidence}</span>
      </td>
      <td style="font-size: 0.8rem; color: var(--text-dim);">${action}</td>
      <td>
        <button class="btn btn-primary" style="padding: 0.3rem 0.65rem; font-size: 0.75rem;" onclick="inspectConsist('${item.file_id}')">
          Inspect Consist 🔬
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

window.inspectConsist = function(caseId) {
  activeCaseId = caseId;
  const selector = document.getElementById('acvCaseSelector');
  if (selector) {
    let found = false;
    for (let opt of selector.options) {
      if (opt.value === caseId) {
        selector.value = caseId;
        found = true;
        break;
      }
    }
    if (!found) {
      const opt = document.createElement('option');
      opt.value = caseId;
      opt.textContent = caseId;
      selector.appendChild(opt);
      selector.value = caseId;
    }
  }
  switchTab('pane-twin');
};

// 3. 8-Car Consist Digital Twin & Visualizer
function initTwinControls() {
  const selector = document.getElementById('acvCaseSelector');
  if (selector) {
    selector.addEventListener('change', (e) => {
      activeCaseId = e.target.value;
      renderConsistTwin();
    });
  }

  const btnInject = document.getElementById('btnInjectLeak');
  if (btnInject) {
    btnInject.addEventListener('click', () => {
      // Pick random car to be faulty
      const randomCarNum = Math.floor(Math.random() * 8) + 1;
      const faultyCarStr = randomCarNum < 10 ? `0${randomCarNum}` : `${randomCarNum}`;
      
      // Generate sequence
      const allCars = ['01','02','03','04','05','06','07','08'].filter(c => c !== faultyCarStr);
      // shuffle rest
      allCars.sort(() => Math.random() - 0.5);
      const newRankStr = [faultyCarStr, ...allCars].join('|');

      CASE_CONFIGS['custom_leak'] = {
        faultyCar: faultyCarStr,
        rankStr: newRankStr,
        desc: `Simulated Leak Injection on Car ${faultyCarStr}`
      };

      activeCaseId = 'custom_leak';
      renderConsistTwin();
    });
  }

  renderConsistTwin();
}

function renderConsistTwin() {
  const grid = document.getElementById('acvConsistGrid');
  if (!grid) return;

  const cfg = CASE_CONFIGS[activeCaseId] || CASE_CONFIGS['case_01'];
  const faultyCarId = cfg.faultyCar;
  const rankList = cfg.rankStr.split('|');

  // Update ranking string display
  const rankDisplay = document.getElementById('acvRankedSequenceDisplay');
  if (rankDisplay) {
    rankDisplay.innerHTML = rankList.map((c, i) => {
      if (i === 0) return `<span style="color: #ef4444; font-weight: 800; background: rgba(239,68,68,0.2); padding: 0.1rem 0.4rem; border-radius: 4px;">Rank 1: Car ${c} (LEAK)</span>`;
      return `Car ${c}`;
    }).join(' <span style="color: var(--text-dim);">&gt;</span> ');
  }

  // Render 8 Cars
  grid.innerHTML = '';
  const carNumbers = ['01', '02', '03', '04', '05', '06', '07', '08'];

  carNumbers.forEach(carNum => {
    const isFaulty = carNum === faultyCarId;
    const rankIndex = rankList.indexOf(carNum) + 1;

    const tempReturn = isFaulty ? 27.4 : (23.0 + (Math.random() * 0.8 - 0.4)).toFixed(1);
    const tempSupply = isFaulty ? 24.8 : (16.5 + (Math.random() * 0.6 - 0.3)).toFixed(1);
    const suctionPress = isFaulty ? '185 kPa' : '395 kPa';
    const subcoolMargin = isFaulty ? '1.1°C' : '6.8°C';

    const card = document.createElement('div');
    card.className = `acv-car-card ${isFaulty ? 'faulty' : 'nominal'}`;

    card.innerHTML = `
      <div class="acv-rank-pill">#${rankIndex}</div>
      <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">${isFaulty ? '🚨' : '🚃'}</div>
      <div style="font-weight: 800; font-size: 0.95rem; color: ${isFaulty ? '#ef4444' : '#fff'}; font-family: var(--font-mono);">
        CAR ${carNum}
      </div>
      <div style="font-size: 0.7rem; color: ${isFaulty ? '#ef4444' : 'var(--color-normal)'}; font-weight: 700; margin-bottom: 0.5rem;">
        ${isFaulty ? 'REFRIGERANT LEAK' : 'NOMINAL HVAC'}
      </div>

      <div style="font-size: 0.75rem; text-align: left; background: rgba(0,0,0,0.3); padding: 0.5rem; border-radius: 4px; font-family: var(--font-mono); line-height: 1.4;">
        <div>T_ret: <span style="color: #fff;">${tempReturn}°C</span></div>
        <div>T_sup: <span style="color: ${isFaulty ? '#ef4444' : '#00f0ff'};">${tempSupply}°C</span></div>
        <div>P_suc: <span style="color: ${isFaulty ? '#ef4444' : 'var(--text-muted)'};">${suctionPress}</span></div>
        <div>Subcool: <span style="color: ${isFaulty ? '#ef4444' : 'var(--color-normal)'};">${subcoolMargin}</span></div>
      </div>
    `;

    grid.appendChild(card);
  });

  // Render Temperature Differential Bars
  const barsContainer = document.getElementById('tempDiffBars');
  if (barsContainer) {
    barsContainer.innerHTML = '';
    carNumbers.forEach(carNum => {
      const isFaulty = carNum === faultyCarId;
      const deltaT = isFaulty ? 2.6 : 6.8;
      const barPct = Math.round((deltaT / 8.0) * 100);
      const barColor = isFaulty ? '#ef4444' : '#10b981';

      const row = document.createElement('div');
      row.style.display = 'flex';
      row.style.alignItems = 'center';
      row.style.gap = '0.75rem';
      row.style.fontSize = '0.8rem';
      row.style.fontFamily = 'var(--font-mono)';

      row.innerHTML = `
        <span style="width: 55px; color: ${isFaulty ? '#ef4444' : 'var(--text-muted)'}; font-weight: 700;">Car ${carNum}</span>
        <div style="flex-grow: 1; height: 6px; background: rgba(255,255,255,0.08); border-radius: 3px; overflow: hidden;">
          <div style="width: ${barPct}%; height: 100%; background: ${barColor}; border-radius: 3px;"></div>
        </div>
        <span style="width: 50px; text-align: right; color: ${barColor}; font-weight: 700;">Δ${deltaT.toFixed(1)}°C</span>
      `;
      barsContainer.appendChild(row);
    });
  }
}
