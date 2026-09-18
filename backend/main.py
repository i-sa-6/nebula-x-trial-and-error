import csv
import io
import os
import sys
import tempfile
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from door_segmentation import segment_door_data
from model.feature_extractor import extract_cycle_features
from model.predict import load_telemetry_rows, get_segments_from_file_or_data

app = FastAPI(title="Rail Vehicle Train Condition Monitoring API")

# Enable CORS for frontend local development (support localhost and 127.0.0.1 on any port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(BASE_DIR, "model", "door_model.joblib")


@app.get("/")
def root():
    return {
        "message": "Rail Condition Monitoring API is running.",
        "model_loaded": os.path.exists(MODEL_PATH),
    }


@app.post("/api/predict/door")
async def predict_door(
    file: UploadFile = File(...),
    segments_file: Optional[UploadFile] = File(None),
):
    """Process uploaded continuous door telemetry CSV and return cycle-by-cycle predictions."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=500,
            detail="Trained model artifact not found. Please train model first.",
        )

    try:
        # Read file contents into memory
        telemetry_bytes = await file.read()
        telemetry_text = telemetry_bytes.decode("utf-8", errors="replace")
        telemetry_reader = csv.DictReader(io.StringIO(telemetry_text))
        telemetry_rows = list(telemetry_reader)

        if not telemetry_rows:
            raise HTTPException(status_code=400, detail="Uploaded telemetry file is empty.")

        # Check optional segments file
        segments_csv_path = None
        if segments_file and hasattr(segments_file, "filename") and segments_file.filename and segments_file.filename.endswith(".csv"):
            seg_bytes = await segments_file.read()
            seg_text = seg_bytes.decode("utf-8", errors="replace")
            temp_seg = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8")
            temp_seg.write(seg_text)
            temp_seg.close()
            segments_csv_path = temp_seg.name

        # Segment data
        cycle_slices = get_segments_from_file_or_data(telemetry_rows, segments_csv_path)

        if segments_csv_path and os.path.exists(segments_csv_path):
            try:
                os.remove(segments_csv_path)
            except OSError:
                pass

        if not cycle_slices:
            raise HTTPException(status_code=400, detail="No door movement cycles detected in the uploaded file.")

        # Load model pipeline
        model_artifact = joblib.load(MODEL_PATH)
        pipeline = model_artifact["pipeline"]
        target_mapping = model_artifact.get(
            "target_mapping", {0: "Normal", 1: "Abnormal resistance"}
        )

        # Feature extraction and prediction
        features_list = []
        cycle_details = []

        for idx, seg_rows in enumerate(cycle_slices):
            if not seg_rows:
                continue
            start_time = seg_rows[0]["Datetime"]
            end_time = seg_rows[-1]["Datetime"]
            n_rows = len(seg_rows)
            duration_sec = round(n_rows * 0.02, 2)

            feats = extract_cycle_features(seg_rows)
            features_list.append(feats)

            operation = "Open" if feats["is_open_op"] == 1.0 else "Close"

            cycle_details.append({
                "segment_id": f"seg_{idx + 1:03d}",
                "start_time": start_time,
                "end_time": end_time,
                "operation": operation,
                "n_rows": n_rows,
                "duration_sec": duration_sec,
                "mean_current_ma": round(feats["cur_mean"], 1),
                "mid_current_ma": round(feats["mid_cur_mean"], 1),
            })

        df_X = pd.DataFrame(features_list)
        y_pred = pipeline.predict(df_X)

        # Build results
        results = []
        normal_count = 0
        abnormal_count = 0

        csv_output = io.StringIO()
        csv_writer = csv.DictWriter(csv_output, fieldnames=["start_time", "end_time", "prediction"])
        csv_writer.writeheader()

        for details, pred_int in zip(cycle_details, y_pred):
            pred_label = target_mapping.get(pred_int, "Normal")
            if pred_label == "Abnormal resistance":
                abnormal_count += 1
            else:
                normal_count += 1

            details["prediction"] = pred_label
            results.append(details)

            csv_writer.writerow({
                "start_time": details["start_time"],
                "end_time": details["end_time"],
                "prediction": pred_label,
            })

        total_cycles = len(results)
        fault_rate_pct = (
            round((abnormal_count / total_cycles) * 100, 1) if total_cycles > 0 else 0.0
        )

        return {
            "filename": file.filename,
            "summary": {
                "total_cycles": total_cycles,
                "normal_count": normal_count,
                "abnormal_count": abnormal_count,
                "fault_rate_pct": fault_rate_pct,
            },
            "segments": results,
            "csv_content": csv_output.getvalue(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing door data: {str(e)}")