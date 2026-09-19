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
  renderVulnerabilitySchematic();
  renderTopFailurePoints();
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

// ==============================================================================
// 8-Car Fleet Vulnerability & Failure Point Mapping
// ==============================================================================

const FLEET_WHEEL_RISK_DATA = [
  // Car 1 (Leading Head Car)
  { w: 1, c: 1, p: 1, s: 'Side I (Left)', e: 1, mx: 1.258, md: 0.518, m: 0.362, t: 'moderate', desc: 'Bogie 1 leading axle; curve entry friction and dynamic hunting' },
  { w: 2, c: 1, p: 2, s: 'Side II (Right)', e: 3, mx: 1.532, md: 0.659, m: 0.382, t: 'critical', desc: 'CRITICAL FAILURE POINT: Right rail lead contact; 3 severe excursions (>1.20 m/s²)' },
  { w: 3, c: 1, p: 3, s: 'Side I (Left)', e: 0, mx: 1.054, md: 0.442, m: 0.288, t: 'moderate', desc: 'Bogie 1 trailing axle; intermediate dynamic load' },
  { w: 4, c: 1, p: 4, s: 'Side II (Right)', e: 3, mx: 1.794, md: 0.642, m: 0.404, t: 'critical', desc: 'CRITICAL FAILURE POINT: Severe right rail shock; peak 1.794 m/s² under braking' },
  { w: 5, c: 1, p: 5, s: 'Side I (Left)', e: 0, mx: 1.121, md: 0.456, m: 0.312, t: 'moderate', desc: 'Bogie 2 leading axle; moderate track transition wear' },
  { w: 6, c: 1, p: 6, s: 'Side II (Right)', e: 1, mx: 1.948, md: 0.531, m: 0.325, t: 'critical', desc: 'CRITICAL PEAK: Transient high-amplitude impact reaching 1.948 m/s²' },
  { w: 7, c: 1, p: 7, s: 'Side I (Left)', e: 1, mx: 1.215, md: 0.485, m: 0.334, t: 'moderate', desc: 'Bogie 2 trailing axle; moderate acoustic roughness' },
  { w: 8, c: 1, p: 8, s: 'Side II (Right)', e: 2, mx: 2.053, md: 0.607, m: 0.378, t: 'critical', desc: 'CRITICAL FAILURE POINT: Car 1 trailing bogie right wheel; peak 2.053 m/s²' },

  // Car 2
  { w: 9, c: 2, p: 1, s: 'Side I (Left)', e: 1, mx: 1.284, md: 0.485, m: 0.315, t: 'moderate', desc: 'Normal curve negotiation excitation' },
  { w: 10, c: 2, p: 2, s: 'Side II (Right)', e: 1, mx: 1.292, md: 0.492, m: 0.318, t: 'moderate', desc: 'Slight right-rail elevation' },
  { w: 11, c: 2, p: 3, s: 'Side I (Left)', e: 0, mx: 0.985, md: 0.380, m: 0.285, t: 'baseline', desc: 'Smooth rolling baseline; zero structural fatigue' },
  { w: 12, c: 2, p: 4, s: 'Side II (Right)', e: 1, mx: 1.341, md: 0.510, m: 0.332, t: 'moderate', desc: 'Intermediate motor bogie dynamic load' },
  { w: 13, c: 2, p: 5, s: 'Side I (Left)', e: 2, mx: 1.617, md: 0.651, m: 0.395, t: 'high', desc: 'HIGH VULNERABILITY: Multiple excursions up to 1.617 m/s² on Left Rail' },
  { w: 14, c: 2, p: 6, s: 'Side II (Right)', e: 0, mx: 1.105, md: 0.415, m: 0.298, t: 'moderate', desc: 'Stable right-rail baseline' },
  { w: 15, c: 2, p: 7, s: 'Side I (Left)', e: 1, mx: 1.350, md: 0.525, m: 0.340, t: 'moderate', desc: 'Bogie trailing oscillation' },
  { w: 16, c: 2, p: 8, s: 'Side II (Right)', e: 1, mx: 1.265, md: 0.485, m: 0.320, t: 'moderate', desc: 'Normal inter-car coupling interaction' },

  // Car 3
  { w: 17, c: 3, p: 1, s: 'Side I (Left)', e: 2, mx: 1.532, md: 0.652, m: 0.385, t: 'high', desc: 'HIGH VULNERABILITY: Sustained curve wear on Left Rail (1.532 m/s²)' },
  { w: 18, c: 3, p: 2, s: 'Side II (Right)', e: 1, mx: 1.310, md: 0.512, m: 0.330, t: 'moderate', desc: 'Right rail leading axle excitation' },
  { w: 19, c: 3, p: 3, s: 'Side I (Left)', e: 0, mx: 1.025, md: 0.410, m: 0.295, t: 'baseline', desc: 'Clean rolling contact' },
  { w: 20, c: 3, p: 4, s: 'Side II (Right)', e: 1, mx: 1.420, md: 0.550, m: 0.355, t: 'moderate', desc: 'Elevated vibration under curve transition' },
  { w: 21, c: 3, p: 5, s: 'Side I (Left)', e: 1, mx: 1.385, md: 0.535, m: 0.345, t: 'moderate', desc: 'Moderate left-rail creepage' },
  { w: 22, c: 3, p: 6, s: 'Side II (Right)', e: 0, mx: 1.140, md: 0.420, m: 0.305, t: 'moderate', desc: 'Within healthy operational limits' },
  { w: 23, c: 3, p: 7, s: 'Side I (Left)', e: 1, mx: 1.410, md: 0.540, m: 0.360, t: 'moderate', desc: 'Leading-trailing load transfer' },
  { w: 24, c: 3, p: 8, s: 'Side II (Right)', e: 1, mx: 1.465, md: 0.575, m: 0.370, t: 'high', desc: 'Approaching threshold limits under dynamic braking' },

  // Car 4 (Mid-Train Motor Car - Highest Risk Consist Node)
  { w: 25, c: 4, p: 1, s: 'Side I (Left)', e: 1, mx: 1.450, md: 0.585, m: 0.380, t: 'high', desc: 'Bogie 1 entry into mid-consist harmonic zone' },
  { w: 26, c: 4, p: 2, s: 'Side II (Right)', e: 1, mx: 1.390, md: 0.540, m: 0.365, t: 'moderate', desc: 'Mid-train draft force response' },
  { w: 27, c: 4, p: 3, s: 'Side I (Left)', e: 3, mx: 1.901, md: 0.703, m: 0.412, t: 'critical', desc: 'CRITICAL POINT OF FAILURE #1: 3 severe excursions; max RMS 1.901 m/s²; primary Left Rail wear site' },
  { w: 28, c: 4, p: 4, s: 'Side II (Right)', e: 2, mx: 1.498, md: 0.573, m: 0.395, t: 'high', desc: 'HIGH VULNERABILITY: Multiple severe Right Rail shocks' },
  { w: 29, c: 4, p: 5, s: 'Side I (Left)', e: 1, mx: 1.415, md: 0.560, m: 0.375, t: 'moderate', desc: 'Secondary bogie pivot resonance' },
  { w: 30, c: 4, p: 6, s: 'Side II (Right)', e: 0, mx: 1.180, md: 0.460, m: 0.320, t: 'moderate', desc: 'Controlled baseline response' },
  { w: 31, c: 4, p: 7, s: 'Side I (Left)', e: 3, mx: 1.593, md: 0.628, m: 0.412, t: 'critical', desc: 'CRITICAL FAILURE POINT: 3 severe excursions; continuous periodic Left Rail excitation' },
  { w: 32, c: 4, p: 8, s: 'Side II (Right)', e: 2, mx: 2.134, md: 0.517, m: 0.403, t: 'critical', desc: 'CRITICAL POINT OF FAILURE #2: FLEET MAXIMUM PEAK (2.134 m/s²); extreme buffing shear' },

  // Car 5
  { w: 33, c: 5, p: 1, s: 'Side I (Left)', e: 1, mx: 1.310, md: 0.485, m: 0.320, t: 'moderate', desc: 'Post-hinge dampened response' },
  { w: 34, c: 5, p: 2, s: 'Side II (Right)', e: 1, mx: 1.250, md: 0.460, m: 0.310, t: 'moderate', desc: 'Moderate right-rail acoustic noise' },
  { w: 35, c: 5, p: 3, s: 'Side I (Left)', e: 0, mx: 0.950, md: 0.360, m: 0.270, t: 'baseline', desc: 'Stable isolated wheelset; healthy baseline' },
  { w: 36, c: 5, p: 4, s: 'Side II (Right)', e: 1, mx: 1.285, md: 0.480, m: 0.325, t: 'moderate', desc: 'Acceptable dynamic compliance' },
  { w: 37, c: 5, p: 5, s: 'Side I (Left)', e: 1, mx: 1.295, md: 0.490, m: 0.330, t: 'moderate', desc: 'Normal passenger service wear' },
  { w: 38, c: 5, p: 6, s: 'Side II (Right)', e: 1, mx: 1.270, md: 0.470, m: 0.315, t: 'moderate', desc: 'Standard rolling contact' },
  { w: 39, c: 5, p: 7, s: 'Side I (Left)', e: 0, mx: 1.010, md: 0.380, m: 0.285, t: 'baseline', desc: 'Smooth track interaction' },
  { w: 40, c: 5, p: 8, s: 'Side II (Right)', e: 1, mx: 1.340, md: 0.495, m: 0.335, t: 'moderate', desc: 'Mild hunting oscillation' },

  // Car 6
  { w: 41, c: 6, p: 1, s: 'Side I (Left)', e: 1, mx: 1.320, md: 0.490, m: 0.325, t: 'moderate', desc: 'Bogie 1 leading wheel excitation' },
  { w: 42, c: 6, p: 2, s: 'Side II (Right)', e: 1, mx: 1.290, md: 0.475, m: 0.315, t: 'moderate', desc: 'Right rail steady rolling' },
  { w: 43, c: 6, p: 3, s: 'Side I (Left)', e: 1, mx: 1.260, md: 0.465, m: 0.305, t: 'moderate', desc: 'Normal vibration profile' },
  { w: 44, c: 6, p: 4, s: 'Side II (Right)', e: 1, mx: 1.330, md: 0.505, m: 0.330, t: 'moderate', desc: 'Mild curve response' },
  { w: 45, c: 6, p: 5, s: 'Side I (Left)', e: 1, mx: 1.345, md: 0.515, m: 0.340, t: 'moderate', desc: 'Trailing bogie excitation' },
  { w: 46, c: 6, p: 6, s: 'Side II (Right)', e: 1, mx: 1.280, md: 0.470, m: 0.310, t: 'moderate', desc: 'Standard operating wear' },
  { w: 47, c: 6, p: 7, s: 'Side I (Left)', e: 1, mx: 1.315, md: 0.485, m: 0.325, t: 'moderate', desc: 'Controlled track contact' },
  { w: 48, c: 6, p: 8, s: 'Side II (Right)', e: 1, mx: 1.370, md: 0.520, m: 0.345, t: 'moderate', desc: 'Dynamic braking load transfer' },

  // Car 7
  { w: 49, c: 7, p: 1, s: 'Side I (Left)', e: 1, mx: 1.350, md: 0.510, m: 0.335, t: 'moderate', desc: 'Bogie 1 leading axle curve response' },
  { w: 50, c: 7, p: 2, s: 'Side II (Right)', e: 1, mx: 1.320, md: 0.495, m: 0.320, t: 'moderate', desc: 'Moderate right-rail vibration' },
  { w: 51, c: 7, p: 3, s: 'Side I (Left)', e: 0, mx: 1.050, md: 0.395, m: 0.280, t: 'baseline', desc: 'Clean baseline rolling' },
  { w: 52, c: 7, p: 4, s: 'Side II (Right)', e: 1, mx: 1.365, md: 0.525, m: 0.340, t: 'moderate', desc: 'Elevated curve slip energy' },
  { w: 53, c: 7, p: 5, s: 'Side I (Left)', e: 1, mx: 1.390, md: 0.535, m: 0.350, t: 'moderate', desc: 'Intermediate load transfer' },
  { w: 54, c: 7, p: 6, s: 'Side II (Right)', e: 1, mx: 1.295, md: 0.480, m: 0.315, t: 'moderate', desc: 'Standard dynamic index' },
  { w: 55, c: 7, p: 7, s: 'Side I (Left)', e: 1, mx: 1.410, md: 0.550, m: 0.360, t: 'high', desc: 'HIGH VULNERABILITY: Pre-tail car load concentration' },
  { w: 56, c: 7, p: 8, s: 'Side II (Right)', e: 1, mx: 1.440, md: 0.565, m: 0.365, t: 'high', desc: 'Dynamic right-rail excursion' },

  // Car 8 (Trailing Tail Car - High Yaw Whip & Corrugation Severity)
  { w: 57, c: 8, p: 1, s: 'Side I (Left)', e: 2, mx: 1.710, md: 0.749, m: 0.435, t: 'critical', desc: 'CRITICAL POINT OF FAILURE #3: Highest defect mean RMS (0.749 m/s²); acute Left Rail corrugation' },
  { w: 58, c: 8, p: 2, s: 'Side II (Right)', e: 2, mx: 1.307, md: 0.670, m: 0.410, t: 'high', desc: 'HIGH VULNERABILITY: Elevated right rail wear under high speed' },
  { w: 59, c: 8, p: 3, s: 'Side I (Left)', e: 1, mx: 1.380, md: 0.530, m: 0.340, t: 'moderate', desc: 'Bogie 1 trailing axle resonance' },
  { w: 60, c: 8, p: 4, s: 'Side II (Right)', e: 1, mx: 1.425, md: 0.555, m: 0.355, t: 'moderate', desc: 'Right rail trailing bogie interaction' },
  { w: 61, c: 8, p: 5, s: 'Side I (Left)', e: 1, mx: 1.450, md: 0.565, m: 0.365, t: 'high', desc: 'Rear bogie leading axle; severe lateral wear' },
  { w: 62, c: 8, p: 6, s: 'Side II (Right)', e: 2, mx: 2.065, md: 0.657, m: 0.420, t: 'critical', desc: 'CRITICAL POINT OF FAILURE #4: Trailing yaw whip peak reaching 2.065 m/s² on Right Rail' },
  { w: 63, c: 8, p: 7, s: 'Side I (Left)', e: 1, mx: 1.395, md: 0.540, m: 0.350, t: 'moderate', desc: 'Tail end structural vibration' },
  { w: 64, c: 8, p: 8, s: 'Side II (Right)', e: 2, mx: 1.685, md: 0.620, m: 0.395, t: 'high', desc: 'HIGH VULNERABILITY: Extreme trailing axle Right Rail excitation' }
];

const TOP_5_FAILURE_POINTS = [
  {
    rank: 1,
    badgeClass: 'badge-rank-1',
    wheelId: 27,
    car: 4,
    pos: 3,
    side: 'Side I (Left Rail)',
    exceedances: '3 of 68 runs',
    maxRms: '1.901 m/s²',
    defectMean: '0.703 m/s²',
    title: 'Car 4 &bull; Wheel 3 (Left)',
    mechanism: 'Mid-train kinematic node buffing shear; primary corrugation initiation site on Left Rail with sustained periodic resonance.'
  },
  {
    rank: 2,
    badgeClass: 'badge-rank-1',
    wheelId: 32,
    car: 4,
    pos: 8,
    side: 'Side II (Right Rail)',
    exceedances: '2 of 68 runs',
    maxRms: '2.134 m/s² 🏆',
    defectMean: '0.517 m/s²',
    title: 'Car 4 &bull; Wheel 8 (Right)',
    mechanism: 'FLEET MAXIMUM RECORD PEAK: Concentrated draft-gear pitch resonance driving extreme wheel unloading and violent track shock.'
  },
  {
    rank: 3,
    badgeClass: 'badge-rank-2',
    wheelId: 57,
    car: 8,
    pos: 1,
    side: 'Side I (Left Rail)',
    exceedances: '2 of 68 runs',
    maxRms: '1.710 m/s²',
    defectMean: '0.749 m/s² 🔥',
    title: 'Car 8 &bull; Wheel 1 (Left)',
    mechanism: 'HIGHEST MEAN SEVERITY: Trailing car leading bogie axle; severe high-amplitude stick-slip friction carving short-pitch corrugation.'
  },
  {
    rank: 4,
    badgeClass: 'badge-rank-2',
    wheelId: 62,
    car: 8,
    pos: 6,
    side: 'Side II (Right Rail)',
    exceedances: '2 of 68 runs',
    maxRms: '2.065 m/s²',
    defectMean: '0.657 m/s²',
    title: 'Car 8 &bull; Wheel 6 (Right)',
    mechanism: 'Trailing tail-whip yaw oscillation (hunting instability); acute Right Rail degradation under elevated track speeds.'
  },
  {
    rank: 5,
    badgeClass: 'badge-rank-other',
    wheelId: 4,
    car: 1,
    pos: 4,
    side: 'Side II (Right Rail)',
    exceedances: '3 of 68 runs',
    maxRms: '1.794 m/s²',
    defectMean: '0.642 m/s²',
    title: 'Car 1 &bull; Wheel 4 (Right)',
    mechanism: 'Lead pilot car curve attack angle; unattenuated collision with rail irregularities under heavy regenerative braking.'
  }
];

function renderVulnerabilitySchematic() {
  const container = document.getElementById('vulnerabilityTrainConsist');
  if (!container) return;
  container.innerHTML = '';

  for (let car = 1; car <= 8; car++) {
    const carElem = document.createElement('div');
    carElem.className = 'train-car';
    carElem.id = `vuln-car-${car}`;

    const header = document.createElement('div');
    header.className = 'car-header';
    header.textContent = car === 1 ? 'CAR 1 (HEAD)' : car === 8 ? 'CAR 8 (TAIL)' : `CAR ${car}`;
    carElem.appendChild(header);

    const wheelsets = document.createElement('div');
    wheelsets.className = 'car-wheelsets';

    // Top Row: Side I (Positions 1, 3, 5, 7)
    const rowTop = document.createElement('div');
    rowTop.className = 'wheel-row';
    [1, 3, 5, 7].forEach(pos => {
      const wheelInfo = FLEET_WHEEL_RISK_DATA.find(w => w.c === car && w.p === pos);
      const node = document.createElement('div');
      node.className = `wheel-node tier-${wheelInfo ? wheelInfo.t : 'baseline'}`;
      node.id = `vuln-wheel-${car}-${pos}`;
      node.textContent = wheelInfo && wheelInfo.t === 'critical' ? '⚠️' : pos;
      node.title = `Car ${car}, Wheel ${pos} (${wheelInfo ? wheelInfo.s : 'Side I'}) - Max RMS: ${wheelInfo ? wheelInfo.mx : 0} m/s²`;
      if (wheelInfo) {
        setupVulnerabilityWheelHover(node, wheelInfo);
      }
      rowTop.appendChild(node);
    });
    wheelsets.appendChild(rowTop);

    // Bottom Row: Side II (Positions 2, 4, 6, 8)
    const rowBottom = document.createElement('div');
    rowBottom.className = 'wheel-row';
    [2, 4, 6, 8].forEach(pos => {
      const wheelInfo = FLEET_WHEEL_RISK_DATA.find(w => w.c === car && w.p === pos);
      const node = document.createElement('div');
      node.className = `wheel-node tier-${wheelInfo ? wheelInfo.t : 'baseline'}`;
      node.id = `vuln-wheel-${car}-${pos}`;
      node.textContent = wheelInfo && wheelInfo.t === 'critical' ? '⚠️' : pos;
      node.title = `Car ${car}, Wheel ${pos} (${wheelInfo ? wheelInfo.s : 'Side II'}) - Max RMS: ${wheelInfo ? wheelInfo.mx : 0} m/s²`;
      if (wheelInfo) {
        setupVulnerabilityWheelHover(node, wheelInfo);
      }
      rowBottom.appendChild(node);
    });
    wheelsets.appendChild(rowBottom);

    carElem.appendChild(wheelsets);
    container.appendChild(carElem);
  }
}

function setupVulnerabilityWheelHover(node, w) {
  const inspectBar = document.getElementById('vulnerabilityInspectBar');
  if (!inspectBar) return;

  const updateInspectBar = () => {
    let tierBadge = '';
    if (w.t === 'critical') tierBadge = '<span style="color:#fca5a5; font-weight:800;">🔴 CRITICAL POINT OF FAILURE</span>';
    else if (w.t === 'high') tierBadge = '<span style="color:#fde68a; font-weight:800;">🟠 HIGH VULNERABILITY</span>';
    else if (w.t === 'moderate') tierBadge = '<span style="color:#fef08a; font-weight:700;">🟡 MODERATE STRESS</span>';
    else tierBadge = '<span style="color:#86efac; font-weight:700;">🟢 BASELINE HEALTHY</span>';

    inspectBar.innerHTML = `<strong>Car ${w.c}, Axle-Box Position ${w.p} (${w.s}):</strong> Tier: ${tierBadge} | Max Test Peak: <strong style="color:var(--primary)">${w.mx.toFixed(3)} m/s²</strong> | Corrugated Mean: <strong>${w.md.toFixed(3)} m/s²</strong> | Runs &gt;1.20 m/s²: <strong>${w.e}/68</strong> &mdash; <em>${w.desc}</em>`;
  };

  node.addEventListener('mouseenter', updateInspectBar);
  node.addEventListener('click', updateInspectBar);
}

function renderTopFailurePoints() {
  const grid = document.getElementById('failurePointsGrid');
  if (!grid) return;
  grid.innerHTML = '';

  TOP_5_FAILURE_POINTS.forEach(item => {
    const card = document.createElement('div');
    card.className = `failure-card ${item.rank === 1 ? 'rank-1' : ''}`;
    card.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.25rem;">
        <span class="failure-rank-badge ${item.badgeClass}">RANK #${item.rank} FAILURE POINT</span>
        <span style="font-size:0.7rem; font-family:var(--font-mono); color:var(--text-dim);">Wheel #${item.wheelId}</span>
      </div>
      <div class="failure-title">
        <span>🚨</span> ${item.title}
      </div>
      <div class="failure-metrics">
        Max Peak: <strong style="color:var(--color-danger); font-size:0.85rem;">${item.maxRms}</strong><br>
        Runs &gt;1.20 m/s²: <strong style="color:var(--text-main);">${item.exceedances}</strong><br>
        Defect Mean: <strong style="color:var(--primary);">${item.defectMean}</strong>
      </div>
      <div class="failure-desc">
        ${item.mechanism}
      </div>
    `;

    // Click highlights wheel on the heatmap
    card.addEventListener('click', () => {
      const wheelNode = document.getElementById(`vuln-wheel-${item.car}-${item.pos}`);
      if (wheelNode) {
        wheelNode.scrollIntoView({ behavior: 'smooth', block: 'center' });
        wheelNode.style.transform = 'scale(1.4)';
        setTimeout(() => { wheelNode.style.transform = ''; }, 1200);
        const inspectBar = document.getElementById('vulnerabilityInspectBar');
        if (inspectBar) {
          inspectBar.innerHTML = `<strong>🚨 FOCUS: Car ${item.car}, Wheel Position ${item.pos} (${item.side}):</strong> Max Peak: <strong style="color:var(--color-danger)">${item.maxRms}</strong> | Exceedances: <strong>${item.exceedances}</strong> &mdash; <em>${item.mechanism}</em>`;
        }
      }
    });

    grid.appendChild(card);
  });
}

