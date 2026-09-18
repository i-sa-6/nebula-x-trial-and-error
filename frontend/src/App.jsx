import React, { useState, useRef } from 'react'
import React, { useState, useRef, useEffect } from 'react'
import './App.css'

function App() {
  const [telemetryFile, setTelemetryFile] = useState(null)
  const [segmentsFile, setSegmentsFile] = useState(null)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [results, setResults] = useState(null)
  const [statusFilter, setStatusFilter] = useState('ALL')
  const [searchQuery, setSearchQuery] = useState('')
  const [backendStatus, setBackendStatus] = useState('checking') // 'online' | 'offline' | 'checking'

  const fileInputRef = useRef(null)
  const segmentsInputRef = useRef(null)

  // Check backend server availability
  const checkBackendHealth = async () => {
    setBackendStatus('checking')
    const candidateUrls = [
      'http://127.0.0.1:8000/',
      'http://localhost:8000/',
      '/',
    ]

    for (const url of candidateUrls) {
      try {
        const res = await fetch(url, { method: 'GET', signal: AbortSignal.timeout(2000) })
        if (res.ok) {
          const data = await res.json().catch(() => ({}))
          if (data.model_loaded !== undefined || data.message) {
            setBackendStatus('online')
            return
          }
        }
      } catch (e) {
        // Continue trying next candidate
      }
    }
    setBackendStatus('offline')
  }

  useEffect(() => {
    checkBackendHealth()
  }, [])

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0]
      if (file.name.endsWith('.csv')) {
        setTelemetryFile(file)
        setError(null)
      } else {
        setError('Please upload a valid .csv file.')
      }
    }
  }

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setTelemetryFile(e.target.files[0])
      setError(null)
    }
  }

  const handleSegmentsSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setSegmentsFile(e.target.files[0])
    }
  }

  const handleAnalyze = async () => {
    if (!telemetryFile) {
      setError('Please select or drag a .csv telemetry file first.')
      return
    }

    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', telemetryFile)
    if (segmentsFile) {
      formData.append('segments_file', segmentsFile)
    }
    // Prepare candidate endpoints to ensure resilience against proxy/port configuration
    const endpoints = [
      'http://127.0.0.1:8000/api/predict/door',
      'http://localhost:8000/api/predict/door',
      '/api/predict/door',
    ]

    try {
      let response
    let lastError = null
    let responseData = null

    for (const endpoint of endpoints) {
      try {
        response = await fetch('/api/predict/door', {
        const formData = new FormData()
        formData.append('file', telemetryFile)
        if (segmentsFile) {
          formData.append('segments_file', segmentsFile)
        }

        const res = await fetch(endpoint, {
          method: 'POST',
          body: formData,
        })
      } catch (proxyErr) {
        // Fallback directly to backend port 8000 in case proxy is not configured or dev server wasn't restarted
        response = await fetch('http://127.0.0.1:8000/api/predict/door', {
          method: 'POST',
          body: formData,
        })
      }

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.detail || `Server error (${response.status})`)
        if (res.ok) {
          responseData = await res.json()
          setBackendStatus('online')
          break
        } else {
          const errData = await res.json().catch(() => ({}))
          lastError = errData.detail || `Server returned error (${res.status}) from ${endpoint}`
        }
      } catch (networkErr) {
        lastError = networkErr.message || 'Connection refused'
      }
    }

      const data = await response.json()
      setResults(data)
    } catch (err) {
      setError(err.message || 'Failed to connect to backend server at http://127.0.0.1:8000. Please ensure backend is running.')
    } finally {
      setLoading(false)
    setLoading(false)

    if (responseData) {
      setResults(responseData)
    } else {
      setBackendStatus('offline')
      setError(
        lastError?.includes('Failed to fetch') || lastError?.includes('refused')
          ? 'Cannot connect to backend server. Please run "start_backend.bat" in the project folder, or run: backend\\.venv\\Scripts\\python.exe -m uvicorn backend.main:app --port 8000'
          : `Prediction failed: ${lastError}`
      )
    }
  }

  const handleDownload = () => {
    if (!results || !results.csv_content) return

    const blob = new Blob([results.csv_content], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.setAttribute('download', 'door_predictions.csv')
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  const resetUpload = () => {
    setTelemetryFile(null)
    setSegmentsFile(null)
    setResults(null)
    setError(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
    if (segmentsInputRef.current) segmentsInputRef.current.value = ''
  }

  // Filtered segments
  const filteredSegments = results?.segments.filter((seg) => {
    const matchesFilter =
      statusFilter === 'ALL' || seg.prediction === statusFilter
    const matchesSearch =
      searchQuery === '' ||
      seg.segment_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      seg.start_time.includes(searchQuery) ||
      seg.end_time.includes(searchQuery) ||
      seg.operation.toLowerCase().includes(searchQuery.toLowerCase())

    return matchesFilter && matchesSearch
  })

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-badge">Subsystem 1 • Condition Monitoring</div>
        <div className="header-top-row">
          <div className="header-badge">Subsystem 1 • Condition Monitoring</div>
          <div className={`status-indicator ${backendStatus}`}>
            <span className="dot"></span>
            {backendStatus === 'online' && 'Backend Connected'}
            {backendStatus === 'offline' && 'Backend Offline'}
            {backendStatus === 'checking' && 'Checking Connection...'}
            {backendStatus === 'offline' && (
              <button
                type="button"
                className="btn-retry-status"
                onClick={checkBackendHealth}
                title="Retry connecting to backend"
              >
                ↻ Retry
              </button>
            )}
          </div>
        </div>

        <h1>Rail Vehicle Door Fault Diagnosis</h1>
        <p className="header-subtitle">
          Temporal cycle segmentation and abnormal mechanical resistance detection from continuous sensor streams.
        </p>
      </header>

      {/* Main Content Area */}
      <main className="app-main">
        {/* Upload Card */}
        <section className="card upload-card">
          <div
            className={`dropzone ${isDragging ? 'dragging' : ''} ${telemetryFile ? 'has-file' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => !telemetryFile && fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              accept=".csv"
              style={{ display: 'none' }}
              onChange={handleFileSelect}
            />

            {!telemetryFile ? (
              <div className="dropzone-content">
                <div className="upload-icon">📂</div>
                <h3>Drag & drop continuous telemetry .csv file</h3>
                <p>or click to browse from your computer (e.g., Test.csv, Train.csv)</p>
                <div className="file-formats-hint">Supports continuous door telemetry stream files</div>
              </div>
            ) : (
              <div className="file-selected-box">
                <div className="file-info-icon">📄</div>
                <div className="file-info-text">
                  <strong>{telemetryFile.name}</strong>
                  <span>{(telemetryFile.size / 1024 / 1024).toFixed(2)} MB</span>
                </div>
                <button
                  type="button"
                  className="btn-text"
                  onClick={(e) => {
                    e.stopPropagation()
                    resetUpload()
                  }}
                >
                  ✕ Remove
                </button>
              </div>
            )}
          </div>

          {/* Optional Segments Dropdown */}
          <div className="advanced-section">
            <button
              type="button"
              className="btn-link"
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              {showAdvanced ? '▼ Hide optional custom segments CSV' : '▶ Optional: Provide pre-segmented CSV (e.g. Train_Segments_Answer.csv)'}
            </button>

            {showAdvanced && (
              <div className="advanced-content">
                <p className="helper-text">
                  Leave empty to automatically detect door opening/closing cycles using the model's adaptive segmentation.
                </p>
                <input
                  type="file"
                  ref={segmentsInputRef}
                  accept=".csv"
                  onChange={handleSegmentsSelect}
                />
                {segmentsFile && (
                  <span className="file-tag">Selected: {segmentsFile.name}</span>
                )}
              </div>
            )}
          </div>

          {/* Action button */}
          <div className="action-row">
            <button
              type="button"
              className="btn-primary"
              disabled={!telemetryFile || loading}
              onClick={handleAnalyze}
            >
              {loading ? (
                <>
                  <span className="spinner"></span> Running Segmentation & Diagnosis...
                </>
              ) : (
                'Run Diagnosis & Predict'
              )}
            </button>
          </div>

          {error && <div className="error-banner">⚠️ {error}</div>}
        </section>

        {/* Results Section */}
        {results && (
          <section className="results-section">
            {/* Summary Cards */}
            <div className="metrics-grid">
              <div className="metric-card">
                <span className="metric-label">Total Cycles Found</span>
                <span className="metric-value">{results.summary.total_cycles}</span>
                <span className="metric-sub">Complete open/close cycles</span>
              </div>

              <div className="metric-card card-normal">
                <span className="metric-label">Normal Operation</span>
                <span className="metric-value text-success">{results.summary.normal_count}</span>
                <span className="metric-sub">Smooth travel & baseline resistance</span>
              </div>

              <div className="metric-card card-abnormal">
                <span className="metric-label">Abnormal Resistance</span>
                <span className="metric-value text-danger">{results.summary.abnormal_count}</span>
                <span className="metric-sub">High friction / rail obstruction</span>
              </div>

              <div className="metric-card">
                <span className="metric-label">Abnormal Cycle Rate</span>
                <span className="metric-value">{results.summary.fault_rate_pct}%</span>
                <span className="metric-sub">Ratio of faulty cycles</span>
              </div>
            </div>

            {/* Download Bar */}
            <div className="download-bar card">
              <div className="download-info">
                <h3>Prediction Output Ready</h3>
                <p>File contains formatted predictions for all {results.summary.total_cycles} cycles matching the competition specification.</p>
              </div>
              <button
                type="button"
                className="btn-download"
                onClick={handleDownload}
              >
                ⬇ Download door_predictions.csv
              </button>
            </div>

            {/* Results Table Card */}
            <div className="card table-card">
              <div className="table-header">
                <div className="table-title">
                  <h3>Predicted Door Cycles</h3>
                  <span>Analyzed from: {results.filename}</span>
                </div>

                <div className="table-controls">
                  <div className="filter-buttons">
                    <button
                      type="button"
                      className={`filter-btn ${statusFilter === 'ALL' ? 'active' : ''}`}
                      onClick={() => setStatusFilter('ALL')}
                    >
                      All ({results.summary.total_cycles})
                    </button>
                    <button
                      type="button"
                      className={`filter-btn ${statusFilter === 'Normal' ? 'active' : ''}`}
                      onClick={() => setStatusFilter('Normal')}
                    >
                      Normal ({results.summary.normal_count})
                    </button>
                    <button
                      type="button"
                      className={`filter-btn ${statusFilter === 'Abnormal resistance' ? 'active' : ''}`}
                      onClick={() => setStatusFilter('Abnormal resistance')}
                    >
                      Abnormal ({results.summary.abnormal_count})
                    </button>
                  </div>

                  <input
                    type="text"
                    placeholder="Search timestamp / ID..."
                    className="search-input"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                </div>
              </div>

              {/* Table */}
              <div className="table-responsive">
                <table>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Start Timestamp</th>
                      <th>End Timestamp</th>
                      <th>Operation</th>
                      <th>Duration</th>
                      <th>Mid-Stroke Current</th>
                      <th>Prediction</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredSegments && filteredSegments.length > 0 ? (
                      filteredSegments.map((seg, idx) => (
                        <tr key={seg.segment_id || idx}>
                          <td className="cell-id">{seg.segment_id}</td>
                          <td className="cell-time">{seg.start_time}</td>
                          <td className="cell-time">{seg.end_time}</td>
                          <td>
                            <span className={`pill-op ${seg.operation.toLowerCase()}`}>
                              {seg.operation}
                            </span>
                          </td>
                          <td>{seg.duration_sec}s ({seg.n_rows} pts)</td>
                          <td>{seg.mid_current_ma} mA</td>
                          <td>
                            <span
                              className={`badge-status ${
                                seg.prediction === 'Normal'
                                  ? 'badge-normal'
                                  : 'badge-abnormal'
                              }`}
                            >
                              {seg.prediction}
                            </span>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan="7" className="text-center empty-cell">
                          No matching segments found.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        )}
      </main>

      <footer className="app-footer">
        Rail Train Condition Monitoring (CdM) • Subsystem 1: Door Fault Diagnosis
      </footer>
    </div>
  )
}

export default App
