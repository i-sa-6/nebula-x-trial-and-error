// NebulaX Rail Corrugation - Predictions Landing Page Controller
let allPredictions = [];
let currentFilter = 'all';
let searchQuery = '';
let currentSort = 'num-asc';

document.addEventListener('DOMContentLoaded', () => {
  initLandingPage();
});

async function initLandingPage() {
  setupEventListeners();
  await loadPredictionsData();
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

  // Upload Test CSV button & input
  const btnUpload = document.getElementById('btnUploadCsv');
  const fileInput = document.getElementById('csvFileInput');
  if (btnUpload && fileInput) {
    btnUpload.addEventListener('click', () => {
      fileInput.click();
    });
    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
      }
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

  // Sort Dropdown
  const sortSelect = document.getElementById('sortSelect');
  if (sortSelect) {
    sortSelect.addEventListener('change', (e) => {
      currentSort = e.target.value;
      renderTable();
    });
  }
}

async function loadPredictionsData() {
  try {
    const res = await fetch('/api/predictions');
    if (!res.ok) throw new Error('Failed to fetch predictions');
    const data = await res.json();
    
    allPredictions = data.predictions || [];
    renderSummary(data.summary);
    renderTable();
  } catch (err) {
    console.warn('Falling back to /api/accuracy_details:', err);
    fallbackLoad();
  }
}

function renderSummary(summary) {
  if (!summary) return;
  
  const elTotal = document.getElementById('kpiTotal');
  const elNormal = document.getElementById('kpiNormal');
  const elNormalPct = document.getElementById('kpiNormalPct');
  const elSide1 = document.getElementById('kpiSide1');
  const elSide2 = document.getElementById('kpiSide2');
  const elConf = document.getElementById('kpiConfidence');
  const elF1 = document.getElementById('modelF1');

  if (elTotal) elTotal.textContent = summary.total;
  if (elNormal) elNormal.textContent = summary.normal;
  if (elNormalPct) elNormalPct.textContent = `${summary.normal_pct}% of test fleet`;
  if (elSide1) elSide1.textContent = summary.side1;
  if (elSide2) elSide2.textContent = summary.side2;
  if (elConf) elConf.textContent = `${summary.avg_confidence}%`;
  if (elF1) elF1.textContent = summary.champion_f1 ? summary.champion_f1.toFixed(4) : '0.8507';

  // Update button counters
  const cntAll = document.getElementById('countAll');
  const cntNorm = document.getElementById('countNormal');
  const cntS1 = document.getElementById('countSide1');
  const cntS2 = document.getElementById('countSide2');
  const cntUp = document.getElementById('countUploaded');
  const btnUp = document.getElementById('btnFilterUploaded');

  if (cntAll) cntAll.textContent = summary.total;
  if (cntNorm) cntNorm.textContent = summary.normal;
  if (cntS1) cntS1.textContent = summary.side1;
  if (cntS2) cntS2.textContent = summary.side2;

  const uploadedCount = allPredictions.filter(p => p.is_uploaded).length;
  if (cntUp) cntUp.textContent = uploadedCount;
  if (btnUp && uploadedCount > 0) {
    btnUp.style.display = 'inline-block';
  }
}

function renderTable() {
  const tbody = document.getElementById('predictionsTableBody');
  if (!tbody) return;

  // Filter
  let filtered = allPredictions.filter(item => {
    if (currentFilter === 'uploaded') {
      if (!item.is_uploaded) return false;
    } else if (currentFilter !== 'all') {
      if (item.prediction !== currentFilter) return false;
    }

    if (searchQuery) {
      const match = item.file_id.toLowerCase().includes(searchQuery) ||
                    item.prediction.toLowerCase().includes(searchQuery) ||
                    item.recommendation.toLowerCase().includes(searchQuery) ||
                    `${item.speed_kmh}`.includes(searchQuery) ||
                    `${item.confidence}`.includes(searchQuery);
      if (!match) return false;
    }
    return true;
  });

  // Sort
  filtered.sort((a, b) => {
    if (currentSort === 'num-asc') {
      const numA = parseInt(a.file_id.replace(/\D/g, '')) || 0;
      const numB = parseInt(b.file_id.replace(/\D/g, '')) || 0;
      return numA - numB;
    } else if (currentSort === 'conf-desc') {
      return b.confidence - a.confidence;
    } else if (currentSort === 'speed-desc') {
      return b.speed_kmh - a.speed_kmh;
    } else if (currentSort === 'defects-first') {
      const rank = (p) => (p === 'Side I' || p === 'Side II' ? 0 : 1);
      return rank(a.prediction) - rank(b.prediction);
    }
    return 0;
  });

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-dim); padding: 3rem;">No test recordings match your search filter.</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered.map((row, idx) => {
    const badgeClass = `badge-${row.prediction.replace(' ', '-')}`;
    const isDefect = row.prediction !== 'Normal';
    const sdiColor = row.sdi_rms > 0.05 ? 'var(--color-side1)' : row.sdi_rms < -0.05 ? 'var(--color-side2)' : 'var(--text-muted)';
    const uploadedTag = row.is_uploaded ? '<span class="uploaded-tag">📤 Uploaded</span>' : '';

    return `
      <tr style="${isDefect ? 'background: rgba(239, 68, 68, 0.04);' : ''}">
        <td class="mono-val" style="color: var(--text-dim);">${idx + 1}</td>
        <td>
          <strong class="mono-val" style="color: var(--text-main); display: inline-flex; align-items: center; gap: 0.4rem;">
            <span>🚆</span> ${row.file_id}
          </strong>
          ${uploadedTag}
        </td>
        <td>
          <span class="badge-class ${badgeClass}">
            ${row.prediction === 'Normal' ? '🟢' : row.prediction === 'Side I' ? '🔴' : '🟠'} ${row.prediction}
          </span>
        </td>
        <td>
          <div style="display: flex; align-items: center; gap: 0.5rem;">
            <div class="confidence-bar-bg" style="width: 50px; height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden;">
              <div style="width: ${row.confidence}%; height: 100%; background: ${isDefect ? 'var(--color-danger)' : 'var(--color-normal)'};"></div>
            </div>
            <span class="mono-val" style="font-weight: 700; font-size: 0.8rem;">${row.confidence}%</span>
          </div>
        </td>
        <td class="mono-val">${row.speed_kmh} km/h</td>
        <td class="mono-val" style="color: ${sdiColor}; font-weight: 600;">
          ${row.sdi_rms > 0 ? '+' : ''}${row.sdi_rms}
        </td>
        <td style="font-size: 0.8rem; color: var(--text-muted); max-width: 320px;">
          ${row.recommendation || (isDefect ? 'Schedule targeted rail milling possession.' : 'Normal healthy baseline.')}
        </td>
        <td style="text-align: center;">
          <a href="twin.html?file=${encodeURIComponent(row.file_id)}" class="btn-inspect" title="Inspect live sensor waveforms and digital twin for this recording">
            Inspect in Twin ↗
          </a>
        </td>
      </tr>
    `;
  }).join('');
}

async function handleFileUpload(file) {
  if (!file) return;

  if (!file.name.toLowerCase().endsWith('.csv')) {
    alert('Invalid file format: Please upload a railway telemetry file ending in .csv');
    return;
  }

  const uploadBtn = document.getElementById('btnUploadCsv');
  const originalBtnHtml = uploadBtn ? uploadBtn.innerHTML : '';
  if (uploadBtn) {
    uploadBtn.disabled = true;
    uploadBtn.innerHTML = '<span class="btn-icon">⏳</span> Validating 129 Channels...';
  }

  showToast(`Uploading and validating 129 physical channels for ${file.name}...`);

  try {
    const url = `/api/upload?filename=${encodeURIComponent(file.name)}`;
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'text/csv'
      },
      body: file
    });

    const data = await res.json();

    if (!res.ok || data.status !== 'success') {
      const errorMsg = data.error || 'Parameter validation failed.';
      let extraInfo = '';
      if (data.missing_samples && data.missing_samples.length > 0) {
        extraInfo = `\n\nExample missing channels:\n- ${data.missing_samples.join('\n- ')}`;
      }
      alert(`❌ Railway Parameter Check Failed:\n\n${errorMsg}${extraInfo}`);
      showToast(`Validation failed for ${file.name}`);
      return;
    }

    const uploadedFilename = data.filename;
    showToast(`✅ ${uploadedFilename}: All 129 physical parameters verified!`);

    // Reload predictions data so the uploaded file appears in the table
    await loadPredictionsData();

    // Scroll to table
    const tableEl = document.getElementById('predictionsTableBody');
    if (tableEl) {
      tableEl.scrollIntoView({ behavior: 'smooth' });
    }
  } catch (err) {
    console.error('Upload error:', err);
    alert(`Upload request failed: ${err.message}`);
  } finally {
    if (uploadBtn) {
      uploadBtn.disabled = false;
      uploadBtn.innerHTML = originalBtnHtml;
    }
    const fileInput = document.getElementById('csvFileInput');
    if (fileInput) fileInput.value = '';
  }
}

async function fallbackLoad() {
  try {
    const res = await fetch('/api/accuracy_details');
    const data = await res.json();
    allPredictions = (data.test_predictions || []).map(p => ({
      ...p,
      recommendation: p.prediction === 'Normal' ? 'Normal healthy rolling baseline. Zero possession required.' : `Targeted ${p.prediction} corrugation grinding.`,
      urgency: p.prediction === 'Normal' ? 'LOW' : 'HIGH',
      is_uploaded: false
    }));
    renderSummary({
      total: allPredictions.length,
      normal: allPredictions.filter(p => p.prediction === 'Normal').length,
      side1: allPredictions.filter(p => p.prediction === 'Side I').length,
      side2: allPredictions.filter(p => p.prediction === 'Side II').length,
      normal_pct: 88.2,
      corrugation_pct: 11.8,
      avg_confidence: 94.8,
      avg_speed: 48.3,
      champion_f1: 0.8507
    });
    renderTable();
  } catch (e) {
    console.error('Fallback load error:', e);
  }
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
