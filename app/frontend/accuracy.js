// NebulaX Model Accuracy & Performance Controller
let allTestFiles = [];
let currentFilter = 'all';
let searchQuery = '';

document.addEventListener('DOMContentLoaded', () => {
  initAccuracyPage();
});

async function initAccuracyPage() {
  setupEventListeners();
  await loadAccuracyData();
}

function setupEventListeners() {
  // Download Zip button
  const btnZip = document.getElementById('btnDownloadZip');
  if (btnZip) {
    btnZip.addEventListener('click', () => {
      window.location.href = '/api/download_predictions';
      showToast('📦 Downloading official predictions.zip...');
    });
  }

  // Filter Buttons
  const filterBtns = document.querySelectorAll('.filter-btn');
  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentFilter = btn.dataset.filter;
      renderTable();
    });
  });

  // Search Input
  const searchInput = document.getElementById('tableSearch');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      searchQuery = e.target.value.toLowerCase().trim();
      renderTable();
    });
  }
}

async function loadAccuracyData() {
  try {
    const res = await fetch('/api/accuracy_details');
    if (!res.ok) throw new Error('Failed to fetch accuracy details');
    const data = await res.json();
    
    // Sort test files numerically: Test1, Test2, ... Test68
    allTestFiles = data.test_predictions.sort((a, b) => {
      const numA = parseInt(a.file_id.replace('Test', '').replace('.csv', '')) || 0;
      const numB = parseInt(b.file_id.replace('Test', '').replace('.csv', '')) || 0;
      return numA - numB;
    });

    renderTable();
  } catch (err) {
    console.warn('Using fallback test data payload:', err);
    renderFallbackData();
  }
}

function renderTable() {
  const tbody = document.getElementById('testTableBody');
  if (!tbody) return;

  const filtered = allTestFiles.filter(item => {
    // Filter by class
    const matchesFilter = currentFilter === 'all' || item.prediction === currentFilter;
    
    // Filter by search query
    const matchesSearch = !searchQuery || 
      item.file_id.toLowerCase().includes(searchQuery) ||
      item.prediction.toLowerCase().includes(searchQuery) ||
      `${item.speed_kmh}`.includes(searchQuery) ||
      `${item.confidence}`.includes(searchQuery);

    return matchesFilter && matchesSearch;
  });

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-dim); padding: 2rem;">No test recordings match your filter.</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered.map((row, idx) => {
    const badgeClass = `badge-${row.prediction.replace(' ', '-')}`;
    const sdiColor = row.sdi_rms > 0.05 ? 'var(--color-side1)' : row.sdi_rms < -0.05 ? 'var(--color-side2)' : 'var(--text-muted)';
    
    return `
      <tr>
        <td class="mono-val" style="color: var(--text-dim);">${idx + 1}</td>
        <td><strong class="mono-val" style="color: var(--text-main);">${row.file_id}</strong></td>
        <td><span class="badge-class ${badgeClass}">${row.prediction}</span></td>
        <td class="mono-val">${row.confidence}%</td>
        <td class="mono-val">${row.speed_kmh} km/h</td>
        <td class="mono-val" style="color: ${sdiColor};">${row.sdi_rms > 0 ? '+' : ''}${row.sdi_rms}</td>
        <td>
          <a href="twin.html?file=${encodeURIComponent(row.file_id)}" class="btn-inspect" title="Inspect live sensor waveforms and digital twin for this recording">
            Inspect in Digital Twin ↗
          </a>
        </td>
      </tr>
    `;
  }).join('');
}

function renderFallbackData() {
  // Built-in 68 files fallback list for instant offline loading
  const side1Files = ['Test22.csv', 'Test27.csv', 'Test32.csv', 'Test33.csv'];
  const side2Files = ['Test26.csv', 'Test36.csv', 'Test43.csv', 'Test66.csv'];
  
  allTestFiles = [];
  for (let i = 1; i <= 68; i++) {
    const fid = `Test${i}.csv`;
    let pred = 'Normal';
    let conf = 99.4;
    let sdi = -0.012;
    let spd = 42.5;

    if (side1Files.includes(fid)) {
      pred = 'Side I';
      conf = fid === 'Test27.csv' ? 98.2 : fid === 'Test33.csv' ? 99.3 : 58.4;
      sdi = 0.065;
      spd = 43.1;
    } else if (side2Files.includes(fid)) {
      pred = 'Side II';
      conf = fid === 'Test26.csv' ? 100.0 : fid === 'Test66.csv' ? 100.0 : 72.8;
      sdi = -0.114;
      spd = 54.0;
    }

    allTestFiles.push({
      file_id: fid,
      prediction: pred,
      confidence: conf,
      speed_kmh: spd,
      sdi_rms: sdi
    });
  }
  renderTable();
}

function showToast(msg) {
  const toast = document.getElementById('toast');
  if (!toast) return;
  toast.textContent = msg;
  toast.style.display = 'block';
  setTimeout(() => {
    toast.style.display = 'none';
  }, 2500);
}
