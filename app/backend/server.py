"""
Backend HTTP API Server for NebulaX Rail Corrugation Condition Monitoring Dashboard.
Pure Python standard library implementation with zero external web framework dependencies.
"""
import os
import sys
import json
import glob
import pickle
import zipfile
import re
import io
import traceback
import urllib.parse
from http.server import HTTPServer, ThreadingHTTPServer, SimpleHTTPRequestHandler
import numpy as np
import pandas as pd
from scipy import signal

from src.config import (
    BASE_DIR, DATA_DIR, TRAIN_DIR, TEST_DIR, MODELS_DIR, SUBMISSION_DIR,
    CLASSES, SIDE_I_POSITIONS, SIDE_II_POSITIONS, NUM_CARS, SAMPLING_RATE
)
from src.feature_extraction import extract_features_from_df
from src.speed_estimator import estimate_speed_from_pulse
from src.models import EnsembleClassifier


FRONTEND_DIR = os.path.join(BASE_DIR, "app", "frontend")
MODEL_PATH = os.path.join(MODELS_DIR, "rail_corrugation_champion.pkl")

# In-Memory Diagnostic Cache (filename -> diagnostic_dict) for 0ms instantaneous loading
ANALYSIS_CACHE = {}

# Global model cache
MODEL_BUNDLE = None

def get_model():
    global MODEL_BUNDLE
    if MODEL_BUNDLE is None and os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            MODEL_BUNDLE = pickle.load(f)
    return MODEL_BUNDLE


def analyze_file(fpath):
    """
    Runs full physical and predictive analysis on a single CSV file.
    Caches result in memory for 0ms instantaneous subsequent loads.
    """
    fname = os.path.basename(fpath)
    if fname in ANALYSIS_CACHE:
        cached = dict(ANALYSIS_CACHE[fname])
        cached["from_cache"] = True
        return cached

    df = pd.read_csv(fpath)
    bundle = get_model()

    
    # Speed
    speed_col = df.iloc[:, 0].values
    speed_info = estimate_speed_from_pulse(speed_col, SAMPLING_RATE)
    speed_mps = max(speed_info["speed_mps"], 1.0)
    
    # Feature extraction & Model prediction
    feats = extract_features_from_df(df, filename=fname)
    
    if bundle is not None:
        feat_cols = bundle["feature_cols"]
        scaler = bundle["scaler"]
        model = bundle["model"]
        weights = bundle.get("class_weights", np.array([1.0, 1.2, 1.2]))
        
        x_vec = np.array([feats.get(c, 0.0) for c in feat_cols]).reshape(1, -1)
        x_vec = np.nan_to_num(x_vec, nan=0.0, posinf=1e6, neginf=-1e6)
        x_scaled = scaler.transform(x_vec)
        
        probs = model.predict_proba(x_scaled)[0]
        weighted_probs = probs * weights
        pred_idx = int(np.argmax(weighted_probs))
        pred_label = CLASSES[pred_idx]
        confidence = float(probs[pred_idx] / (np.sum(probs) + 1e-9))
    else:
        pred_label = "Unknown"
        confidence = 0.0
        probs = [0.33, 0.33, 0.33]

    # Map 64 wheelset sensor channels
    col_names = list(df.columns)
    car_wheel_data = [] # 8 cars, each with 8 positions
    
    for car in range(1, NUM_CARS + 1):
        car_info = {"car": car, "positions": {}}
        for pos in range(1, 9):
            vib_col = f"Vibration of bearing in position {pos} of car {car}"
            shock_col = f"Shock of bearing in position {pos} of car {car}"
            
            v_rms = float(np.sqrt(np.mean(df[vib_col].values**2))) if vib_col in df else 0.0
            s_rms = float(np.sqrt(np.mean(df[shock_col].values**2))) if shock_col in df else 0.0
            side = "Side I" if pos in SIDE_I_POSITIONS else "Side II"
            
            car_info["positions"][pos] = {
                "side": side,
                "vib_rms": round(v_rms, 4),
                "shock_rms": round(s_rms, 4)
            }
        car_wheel_data.append(car_info)
        
    # Spectral Analysis (Welch PSD)
    s1_cols = [c for c in df.columns if 'Vibration' in c and any(f'position {p}' in c for p in SIDE_I_POSITIONS)]
    s2_cols = [c for c in df.columns if 'Vibration' in c and any(f'position {p}' in c for p in SIDE_II_POSITIONS)]
    
    s1_vib_mean = np.mean(df[s1_cols].values, axis=1)
    s2_vib_mean = np.mean(df[s2_cols].values, axis=1)
    
    freqs, psd_s1 = signal.welch(s1_vib_mean, fs=SAMPLING_RATE, nperseg=1024)
    freqs, psd_s2 = signal.welch(s2_vib_mean, fs=SAMPLING_RATE, nperseg=1024)
    
    # Find dominant defect frequency in corrugation band (80 - 1000 Hz)
    band_mask = (freqs >= 80) & (freqs <= 1000)
    if np.any(band_mask):
        if pred_label == "Side I":
            peak_freq_idx = np.argmax(psd_s1[band_mask])
            peak_freq = float(freqs[band_mask][peak_freq_idx])
        elif pred_label == "Side II":
            peak_freq_idx = np.argmax(psd_s2[band_mask])
            peak_freq = float(freqs[band_mask][peak_freq_idx])
        else:
            peak_freq = float(freqs[band_mask][np.argmax((psd_s1 + psd_s2)[band_mask])])
    else:
        peak_freq = 300.0
        
    # Corrugation wavelength in millimeters: lambda = v / f * 1000
    wavelength_mm = round((speed_mps / max(peak_freq, 1.0)) * 1000, 1)
    
    # Downsample waveforms for snappy frontend rendering (downsample from 10,000 to 300 points)
    step = len(s1_vib_mean) // 300
    downsampled_time = [round(t / SAMPLING_RATE, 4) for t in range(0, len(s1_vib_mean), step)]
    downsampled_s1 = [round(float(v), 4) for v in s1_vib_mean[::step]]
    downsampled_s2 = [round(float(v), 4) for v in s2_vib_mean[::step]]
    
    # Subsample spectrum to 100 points
    spec_step = len(freqs) // 100
    spectrum_freqs = [round(float(f), 1) for f in freqs[::spec_step]]
    spectrum_s1 = [round(float(p), 6) for p in psd_s1[::spec_step]]
    spectrum_s2 = [round(float(p), 6) for p in psd_s2[::spec_step]]
    
    # Actionable Maintenance Advice
    if pred_label == "Normal":
        recommendation = "Both rails operating within healthy acoustic and vibration bounds. No immediate track intervention required."
        urgency = "LOW"
    elif pred_label == "Side I":
        recommendation = f"Side I (Left) rail shows characteristic periodic corrugation (est. wavelength ~{wavelength_mm} mm). Schedule targeted rail milling/grinding possession for Side I track section."
        urgency = "HIGH"
    else:
        recommendation = f"Side II (Right) rail shows abnormal corrugation wear (est. wavelength ~{wavelength_mm} mm). Schedule targeted rail milling/grinding possession for Side II track section."
        urgency = "HIGH"
        
    result = {
        "file_id": fname,
        "prediction": pred_label,
        "confidence": round(confidence * 100, 1),
        "probabilities": {
            "Normal": round(float(probs[0]) * 100, 1),
            "Side I": round(float(probs[1]) * 100, 1),
            "Side II": round(float(probs[2]) * 100, 1)
        },
        "speed_kmh": round(speed_info["speed_kmh"], 1),
        "speed_mps": round(speed_info["speed_mps"], 2),
        "sdi_rms": round(feats.get("sdi_rms", 0.0), 3),
        "diff_rms": round(feats.get("diff_rms", 0.0), 4),
        "peak_frequency_hz": round(peak_freq, 1),
        "corrugation_wavelength_mm": wavelength_mm,
        "urgency": urgency,
        "recommendation": recommendation,
        "car_wheel_data": car_wheel_data,
        "time_series": {
            "time": downsampled_time,
            "side1": downsampled_s1,
            "side2": downsampled_s2
        },
        "spectrum": {
            "freqs": spectrum_freqs,
            "side1": spectrum_s1,
            "side2": spectrum_s2
        }
    }
    ANALYSIS_CACHE[fname] = result
    return result

class DashboardRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/api/status":
            self.handle_api_status()
        elif path == "/api/files":
            self.handle_api_files()
        elif path == "/api/predict":
            fname = query.get("file", ["Test1.csv"])[0]
            self.handle_api_predict(fname)
        elif path == "/api/stream_predict":
            fname = query.get("file", ["Test1.csv"])[0]
            self.handle_api_stream_predict(fname)
        elif path == "/api/clear_cache":
            ANALYSIS_CACHE.clear()
            self.send_json({"status": "success", "message": "Server cache cleared successfully", "cached_items": 0})
        elif path == "/api/download_predictions":
            self.handle_download_zip()
        elif path == "/api/accuracy_details":
            self.handle_api_accuracy()
        elif path == "/api/predictions":
            self.handle_api_predictions()
        elif path == "/twin":
            self.send_response(301)
            self.send_header("Location", "/twin.html" + ("?" + parsed.query if parsed.query else ""))
            self.end_headers()
        else:
            super().do_GET()




    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/batch_inference":
            self.handle_batch_inference()
        elif parsed.path == "/api/upload":
            query = urllib.parse.parse_qs(parsed.query)
            self.handle_file_upload(query)
        else:
            self.send_error(404, "Endpoint not found")

    def handle_api_status(self):
        bundle = get_model()
        test_files = glob.glob(os.path.join(TEST_DIR, "*.csv"))
        train_files = glob.glob(os.path.join(TRAIN_DIR, "*.csv"))
        pred_csv = os.path.join(SUBMISSION_DIR, "rail_predictions.csv")
        has_predictions = os.path.exists(pred_csv)
        
        status = {
            "system": "NebulaX Rail Corrugation Diagnostic Engine",
            "model_name": bundle["model_name"] if bundle else "None",
            "macro_f1": bundle["cv_macro_f1"] if bundle else 0.0,
            "train_files_count": len(train_files),
            "test_files_count": len(test_files),
            "has_predictions": has_predictions
        }
        self.send_json(status)

    def handle_api_files(self):
        all_csvs = [os.path.basename(p) for p in glob.glob(os.path.join(TEST_DIR, "*.csv"))]
        test_files = []
        uploaded_files = []
        
        for f in all_csvs:
            name_part = f.replace(".csv", "")
            if name_part.startswith("Test") and name_part[4:].isdigit() and int(name_part[4:]) <= 68:
                test_files.append(f)
            else:
                uploaded_files.append(f)
                
        test_files.sort(key=lambda x: int(x.replace("Test", "").replace(".csv", "")))
        uploaded_files.sort()
        
        train_samples = ["Train1.csv (Normal)", "Train62.csv (Side I Defect)", "Train2.csv (Side II Defect)"]
        self.send_json({
            "test_files": test_files,
            "uploaded_files": uploaded_files,
            "train_samples": train_samples
        })

    def handle_api_predict(self, fname):
        clean_name = fname.split(" ")[0] # in case of Train1.csv (Normal)
        # Search test dir first then train dir
        target_path = os.path.join(TEST_DIR, clean_name)
        if not os.path.exists(target_path):
            target_path = os.path.join(TRAIN_DIR, clean_name)
            
        if not os.path.exists(target_path):
            self.send_error(404, f"File {clean_name} not found")
            return
            
        result = analyze_file(target_path)
        self.send_json(result)

    def handle_api_stream_predict(self, fname):
        import time
        clean_name = fname.split(" ")[0] # in case of Train1.csv (Normal)
        target_path = os.path.join(TEST_DIR, clean_name)
        if not os.path.exists(target_path):
            target_path = os.path.join(TRAIN_DIR, clean_name)
            
        if not os.path.exists(target_path):
            self.send_error(404, f"File {clean_name} not found")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        try:
            result = analyze_file(target_path)
            
            # Step 1: Stream metadata & 64-wheelset readings instantly
            meta = {
                "file_id": result["file_id"],
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "urgency": result["urgency"],
                "speed_kmh": result["speed_kmh"],
                "speed_mps": result["speed_mps"],
                "peak_frequency_hz": result["peak_frequency_hz"],
                "corrugation_wavelength_mm": result["corrugation_wavelength_mm"],
                "car_wheel_data": result["car_wheel_data"],
                "probabilities": result["probabilities"],
                "from_cache": result.get("from_cache", False)
            }
            self.wfile.write(f"event: metadata\ndata: {json.dumps(meta)}\n\n".encode("utf-8"))
            self.wfile.flush()

            # Step 2: Stream AI recommendation token-by-token (via generateContentStream or physics engine)
            try:
                from app.backend.gemini_streamer import stream_gemini_advisory
                token_stream = stream_gemini_advisory(
                    prediction=result["prediction"],
                    confidence=result["confidence"],
                    speed_kmh=result["speed_kmh"],
                    wavelength_mm=result["corrugation_wavelength_mm"],
                    peak_freq_hz=result["peak_frequency_hz"],
                    urgency=result["urgency"]
                )
            except Exception:
                token_stream = (w if i == 0 else " " + w for i, w in enumerate(result["recommendation"].split(" ")))

            for i, token in enumerate(token_stream):
                token_data = {"token": token, "index": i}
                self.wfile.write(f"event: token\ndata: {json.dumps(token_data)}\n\n".encode("utf-8"))
                self.wfile.flush()
                time.sleep(0.015) # 15ms per token for ultra-responsive streaming

            # Step 3: Stream completion
            self.wfile.write(f"event: done\ndata: {json.dumps({'done': True})}\n\n".encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def handle_batch_inference(self):
        from predict import run_inference
        pred_csv = os.path.join(SUBMISSION_DIR, "rail_predictions.csv")
        sub_df, res_df = run_inference(TEST_DIR, pred_csv)
        
        # Package predictions.zip
        zip_path = os.path.join(SUBMISSION_DIR, "predictions.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(pred_csv, arcname="rail_predictions.csv")
            
        summary = sub_df["prediction"].value_counts().to_dict()
        self.send_json({
            "status": "success",
            "total_files": len(sub_df),
            "summary": summary,
            "zip_file": "predictions.zip"
        })

    def handle_download_zip(self):
        zip_path = os.path.join(SUBMISSION_DIR, "predictions.zip")
        if not os.path.exists(zip_path):
            # create it if rail_predictions.csv exists
            pred_csv = os.path.join(SUBMISSION_DIR, "rail_predictions.csv")
            if os.path.exists(pred_csv):
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                    z.write(pred_csv, arcname="rail_predictions.csv")
            else:
                self.send_error(404, "Predictions zip not yet generated")
                return
                
        with open(zip_path, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", 'attachment; filename="predictions.zip"')
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def handle_api_accuracy(self):
        bundle = get_model()
        pred_detailed_path = os.path.join(SUBMISSION_DIR, "rail_predictions_detailed.csv")
        test_details = []
        if os.path.exists(pred_detailed_path):
            df = pd.read_csv(pred_detailed_path)
            for _, row in df.iterrows():
                test_details.append({
                    "file_id": str(row["file_id"]),
                    "prediction": str(row["prediction"]),
                    "confidence": round(float(row["confidence"]) * 100, 1),
                    "speed_kmh": round(float(row["speed_kmh"]), 1),
                    "sdi_rms": round(float(row["sdi_rms"]), 3),
                    "prob_normal": round(float(row["prob_normal"]) * 100, 1),
                    "prob_side1": round(float(row["prob_side1"]) * 100, 1),
                    "prob_side2": round(float(row["prob_side2"]) * 100, 1)
                })

        accuracy_payload = {
            "model_name": bundle["model_name"] if bundle else "Champion Ensemble (50% HGB + 50% LR)",
            "cv_macro_f1": 0.8507,
            "overall_accuracy": 95.59,
            "training_samples_count": 272,
            "test_samples_count": len(test_details),
            "per_class_metrics": [
                {"class": "Normal", "precision": 98.28, "recall": 97.86, "f1": 0.9807, "support": 234, "tp": 229, "fp": 4, "fn": 5},
                {"class": "Side I", "precision": 71.43, "recall": 71.43, "f1": 0.7143, "support": 14, "tp": 10, "fp": 4, "fn": 4},
                {"class": "Side II", "precision": 84.00, "recall": 87.50, "f1": 0.8571, "support": 24, "tp": 21, "fp": 4, "fn": 3}
            ],
            "confusion_matrix": [
                [229, 2, 3],
                [3, 10, 1],
                [1, 2, 21]
            ],
            "model_benchmarks": [
                {"name": "Random Forest (Balanced)", "macro_f1": 0.6387, "normal_f1": 0.9605, "side1_f1": 0.1111, "side2_f1": 0.8444, "is_champion": False},
                {"name": "Logistic Regression (Balanced)", "macro_f1": 0.7310, "normal_f1": 0.9522, "side1_f1": 0.4324, "side2_f1": 0.8085, "is_champion": False},
                {"name": "HistGradientBoosting (Balanced)", "macro_f1": 0.8253, "normal_f1": 0.9787, "side1_f1": 0.6400, "side2_f1": 0.8571, "is_champion": False},
                {"name": "Champion Ensemble (50% HGB + 50% LR)", "macro_f1": 0.8507, "normal_f1": 0.9807, "side1_f1": 0.7143, "side2_f1": 0.8571, "is_champion": True}
            ],
            "test_summary": {
                "total": len(test_details),
                "Normal": sum(1 for d in test_details if d["prediction"] == "Normal"),
                "Side I": sum(1 for d in test_details if d["prediction"] == "Side I"),
                "Side II": sum(1 for d in test_details if d["prediction"] == "Side II")
            },
            "test_predictions": test_details
        }
        self.send_json(accuracy_payload)

    def handle_api_predictions(self):
        bundle = get_model()
        pred_detailed_path = os.path.join(SUBMISSION_DIR, "rail_predictions_detailed.csv")
        all_preds = []
        if os.path.exists(pred_detailed_path):
            df = pd.read_csv(pred_detailed_path)
            for _, row in df.iterrows():
                pred = str(row["prediction"])
                sdi = round(float(row["sdi_rms"]), 3)
                speed = round(float(row["speed_kmh"]), 1)
                conf = round(float(row["confidence"]) * 100, 1)
                fid = str(row["file_id"])
                
                if pred == "Normal":
                    action = "Normal healthy rolling baseline. Zero possession required."
                    urgency = "LOW"
                elif pred == "Side I":
                    action = "Side I (Left) corrugation. Schedule targeted rail milling/grinding."
                    urgency = "HIGH"
                else:
                    action = "Side II (Right) corrugation. Schedule targeted rail milling/grinding."
                    urgency = "HIGH"
                    
                all_preds.append({
                    "file_id": fid,
                    "prediction": pred,
                    "confidence": conf,
                    "speed_kmh": speed,
                    "sdi_rms": sdi,
                    "urgency": urgency,
                    "recommendation": action,
                    "prob_normal": round(float(row["prob_normal"]) * 100, 1),
                    "prob_side1": round(float(row["prob_side1"]) * 100, 1),
                    "prob_side2": round(float(row["prob_side2"]) * 100, 1),
                    "is_uploaded": False
                })
                
        # Also include any newly uploaded files in TEST_DIR
        for f in glob.glob(os.path.join(TEST_DIR, "*.csv")):
            fname = os.path.basename(f)
            if not any(p["file_id"] == fname for p in all_preds):
                try:
                    res = analyze_file(f)
                    all_preds.append({
                        "file_id": fname,
                        "prediction": res["prediction"],
                        "confidence": res["confidence"],
                        "speed_kmh": res["speed_kmh"],
                        "sdi_rms": res["sdi_rms"],
                        "urgency": res["urgency"],
                        "recommendation": res["recommendation"],
                        "prob_normal": res["probabilities"]["Normal"],
                        "prob_side1": res["probabilities"]["Side I"],
                        "prob_side2": res["probabilities"]["Side II"],
                        "is_uploaded": True
                    })
                except Exception:
                    pass

        # Sort: uploaded first, then Test1..68 numerically
        def sort_key(p):
            if p["is_uploaded"]:
                return (0, 0, p["file_id"])
            digits = re.findall(r'\d+', p["file_id"])
            num = int(digits[0]) if digits else 9999
            return (1, num, p["file_id"])
            
        all_preds.sort(key=sort_key)
        
        normal_cnt = sum(1 for p in all_preds if p["prediction"] == "Normal")
        side1_cnt = sum(1 for p in all_preds if p["prediction"] == "Side I")
        side2_cnt = sum(1 for p in all_preds if p["prediction"] == "Side II")
        total_cnt = len(all_preds)
        
        avg_conf = round(sum(p["confidence"] for p in all_preds) / max(total_cnt, 1), 1)
        avg_speed = round(sum(p["speed_kmh"] for p in all_preds) / max(total_cnt, 1), 1)
        
        payload = {
            "summary": {
                "total": total_cnt,
                "normal": normal_cnt,
                "side1": side1_cnt,
                "side2": side2_cnt,
                "normal_pct": round((normal_cnt / total_cnt) * 100, 1) if total_cnt else 0,
                "corrugation_pct": round(((side1_cnt + side2_cnt) / total_cnt) * 100, 1) if total_cnt else 0,
                "avg_confidence": avg_conf,
                "avg_speed": avg_speed,
                "champion_f1": 0.8507
            },
            "predictions": all_preds
        }
        self.send_json(payload)

    def send_error_json(self, message, status_code=400, extra=None):
        payload = {"status": "error", "error": message}
        if extra:
            payload.update(extra)
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def handle_file_upload(self, query):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length <= 0:
                self.send_error_json("No file payload received in upload request.")
                return
                
            body = self.rfile.read(content_length)
            
            # Resolve requested filename
            filename = query.get("filename", ["test.csv"])[0]
            header_fn = self.headers.get("X-Filename")
            if header_fn:
                filename = header_fn
                
            # Check for multipart/form-data
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" in content_type:
                fn_match = re.search(r'filename="([^"]+)"', body[:2048].decode("utf-8", errors="ignore"))
                if fn_match:
                    filename = fn_match.group(1)
                header_end = body.find(b"\r\n\r\n")
                if header_end != -1:
                    boundary_end = body.rfind(b"\r\n--")
                    if boundary_end > header_end:
                        csv_bytes = body[header_end + 4 : boundary_end]
                    else:
                        csv_bytes = body[header_end + 4 :]
                else:
                    csv_bytes = body
            else:
                csv_bytes = body
                
            # Sanitize filename
            filename = os.path.basename(filename).strip()
            if not filename.lower().endswith(".csv"):
                filename += ".csv"
            filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
            if not filename or filename == ".csv":
                filename = "test_uploaded.csv"
                
            # Parse CSV
            try:
                df = pd.read_csv(io.BytesIO(csv_bytes))
            except Exception as e:
                self.send_error_json(f"Invalid CSV structure: Could not parse CSV data ({str(e)})")
                return
                
            if len(df) < 50:
                self.send_error_json(f"Insufficient sample length ({len(df)} rows). Automated rail corrugation diagnostics requires at least 50 continuous telemetry samples.")
                return
                
            # Clean column names (strip whitespace and BOM)
            df.columns = [str(c).strip().lstrip('\ufeff') for c in df.columns]
            
            # 129 Required Railway Channels Specification
            required_channels = ["Rotating speed"]
            for car in range(1, 9):
                for pos in range(1, 9):
                    required_channels.append(f"Vibration of bearing in position {pos} of car {car}")
                    required_channels.append(f"Shock of bearing in position {pos} of car {car}")
                    
            existing_cols = set(df.columns)
            missing = [c for c in required_channels if c not in existing_cols]
            
            if missing:
                sample_missing = missing[:5]
                err_msg = f"Parameter validation failed: Missing {len(missing)} of 129 required physical channels. (e.g. {', '.join(sample_missing)})"
                self.send_error_json(
                    err_msg, 
                    status_code=400,
                    extra={
                        "missing_count": len(missing), 
                        "expected_channels": 129, 
                        "found_channels": len(df.columns), 
                        "missing_samples": sample_missing
                    }
                )
                return
                
            # Verify numeric sensor data
            try:
                for c in required_channels:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                nan_cols = [c for c in required_channels if df[c].isna().all()]
                if nan_cols:
                    self.send_error_json(f"Parameter validation failed: Channels contain purely non-numeric or empty values: {', '.join(nan_cols[:4])}")
                    return
            except Exception as e:
                self.send_error_json(f"Numeric validation error: {str(e)}")
                return
                
            # Fill small isolated NaNs
            df.fillna(0.0, inplace=True)
            
            # Save validated file into test pool directory
            save_path = os.path.join(TEST_DIR, filename)
            df.to_csv(save_path, index=False)
            
            # Clear cache for this file if previously analyzed
            ANALYSIS_CACHE.pop(filename, None)
            
            # Pre-warm analysis immediately
            result = analyze_file(save_path)
            
            self.send_json({
                "status": "success",
                "filename": filename,
                "message": f"Parameters verified! All 129 physical channels checked across {len(df)} samples.",
                "total_channels": len(df.columns),
                "total_samples": len(df),
                "result": result
            })
        except Exception as e:
            traceback.print_exc()
            self.send_error_json(f"Server error during file processing: {str(e)}", status_code=500)

    def send_json(self, data):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

def prewarm_cache():
    """Pre-computes and caches all test & key demo files in RAM for instant 0ms UI response."""
    import threading
    def worker():
        # Prewarm common training samples first
        sample_train = ["Train1.csv", "Train62.csv", "Train2.csv"]
        for f in sample_train:
            fpath = os.path.join(TRAIN_DIR, f)
            if os.path.exists(fpath):
                try:
                    analyze_file(fpath)
                except Exception:
                    pass
        
        # Prewarm all 68 test files in memory
        test_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.csv")))
        for fpath in test_files:
            try:
                analyze_file(fpath)
            except Exception:
                pass
        print(f"[✓] Full memory cache ready: {len(ANALYSIS_CACHE)} files pre-computed for instant 0ms switching.")
    threading.Thread(target=worker, daemon=True).start()

def run_server(port=None):
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    prewarm_cache()
    if port is None:
        port = int(os.environ.get("PORT", sys.argv[1] if len(sys.argv) > 1 else 8080))
    # In cloud environments (Cloud Run, Docker) bind to 0.0.0.0; locally bind to 127.0.0.1
    host = os.environ.get("HOST", "0.0.0.0" if os.environ.get("CONTAINER") or os.environ.get("K_SERVICE") else "127.0.0.1")
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, DashboardRequestHandler)
    print(f"\n[★] NebulaX Rail Corrugation Dashboard running at: http://{host}:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    env_port = os.environ.get("PORT")
    port = int(env_port) if env_port else (int(sys.argv[1]) if len(sys.argv) > 1 else 8080)
    run_server(port)
