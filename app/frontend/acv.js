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
async function fetchAcvPredictions() {
  try {
    const res = await fetch('/api/acv/predictions');
    const data = await res.json();
    acvData = data;

    renderTable();
  } catch (err) {
    console.error('Failed to fetch ACV predictions:', err);
  }
}

function renderTable() {
  const tbody = document.getElementById('acvTableBody');
  if (!tbody || !acvData || !acvData.cases) return;

  tbody.innerHTML = '';

  acvData.cases.forEach((item, idx) => {
    const tr = document.createElement('tr');
    const caseKey = `case_0${idx + 1}`;
    
    // Split sequence
    const parts = item.ranked_cars.split('|');
    const formattedRank = parts.map((p, i) => i === 0 ? `<strong style="color: #ef4444;">${p}</strong>` : p).join(' &gt; ');

    tr.innerHTML = `
      <td style="font-family: var(--font-mono); font-weight: 700; color: #fff;">${item.file_id}</td>
      <td>
        <span style="display: inline-block; padding: 0.25rem 0.65rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 800; color: #ef4444; background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); font-family: var(--font-mono);">
          ⚠️ ${item.faulty_car}
        </span>
      </td>
      <td style="font-family: var(--font-mono); font-size: 0.85rem; color: #38bdf8;">
        ${formattedRank}
      </td>
      <td style="font-size: 0.85rem; color: var(--text-muted);">${item.symptom}</td>
      <td>
        <span style="font-family: var(--font-mono); font-weight: 700; color: var(--color-normal);">${item.confidence}</span>
      </td>
      <td style="font-size: 0.8rem; color: var(--text-dim);">${item.action}</td>
      <td>
        <button class="btn btn-primary" style="padding: 0.3rem 0.65rem; font-size: 0.75rem;" onclick="inspectConsist('${caseKey}')">
          Inspect Consist 🔬
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

window.inspectConsist = function(caseKey) {
  activeCaseId = caseKey;
  const selector = document.getElementById('acvCaseSelector');
  if (selector) selector.value = caseKey;
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
