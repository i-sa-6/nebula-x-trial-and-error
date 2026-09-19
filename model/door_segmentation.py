"""Door Subsystem Cycle Segmentation and Resistance Classification.

This module provides functions and a command-line interface to segment continuous
rail vehicle door telemetry data (such as Train.csv or Test.csv) into individual
door-opening and door-closing cycles, identify the operation type ('Open' vs 'Close'),
and classify each cycle's health status ('Normal' vs 'Abnormal resistance').
"""

import argparse
import csv
from datetime import datetime
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union


def parse_datetime(ts_str: str) -> datetime:
    """Parse custom hyphen-separated datetime string.

    Format: Year-Month-Date-Hour-Minute-Second-Millisecond
    Example: '2023-7-5-0-0-3-760'
    """
    parts = list(map(int, ts_str.strip().split("-")))
    # Year, Month, Day, Hour, Minute, Second, Millisecond (* 1000 for microseconds)
    return datetime(
        parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6] * 1000
    )


def segment_door_data(
    rows: List[Dict[str, str]],
    gap_threshold_sec: float = 0.5,
) -> List[List[Dict[str, str]]]:
    """Segment a continuous list of door telemetry rows into distinct cycles.

    Segmentation Criteria:
    1. Temporal Gap: Within an active door movement cycle, the door controller
       logs samples at 50 Hz (20 ms interval, dt = 0.02s). When the door is idle
       between cycles, logging is suspended or separated by a dwell gap
       (typically > 10 seconds). Any inter-sample time delta dt > gap_threshold_sec
       indicates a cycle boundary.
    2. Active State Shift: If continuous recording exists without time gaps,
       boundaries are detected when motion switches between Opening and Closing
       or when door returns to rest.

    Args:
        rows: List of row dictionaries from the door CSV.
        gap_threshold_sec: Time gap in seconds to delineate a new cycle (default 0.5s).

    Returns:
        List of segments, where each segment is a list of row dicts for that cycle.
    """
    if not rows:
        return []

    segments: List[List[Dict[str, str]]] = []
    current_segment: List[Dict[str, str]] = [rows[0]]

    for i in range(1, len(rows)):
        prev_row = rows[i - 1]
        curr_row = rows[i]

        t_prev = parse_datetime(prev_row["Datetime"])
        t_curr = parse_datetime(curr_row["Datetime"])
        dt = (t_curr - t_prev).total_seconds()

        # Check for temporal gap (primary criteria for train/test data)
        is_gap_split = dt > gap_threshold_sec

        # Secondary check: operation mode flip (e.g. Opening -> Closing without idle gap)
        prev_is_opening = prev_row.get("Door is opening") == "1"
        curr_is_opening = curr_row.get("Door is opening") == "1"
        prev_is_closing = prev_row.get("Door is closing") == "1"
        curr_is_closing = curr_row.get("Door is closing") == "1"

        is_op_flip = (prev_is_opening and curr_is_closing) or (
            prev_is_closing and curr_is_opening
        )

        if is_gap_split or is_op_flip:
            segments.append(current_segment)
            current_segment = [curr_row]
        else:
            current_segment.append(curr_row)

    if current_segment:
        segments.append(current_segment)

    return segments


def classify_segment(
    seg_rows: List[Dict[str, str]],
    open_current_threshold: float = 270.0,
    close_current_threshold: float = 250.0,
) -> Dict[str, Any]:
    """Analyze a single cycle segment to determine operation, status, and metrics.

    Criteria:
    - Operation ('Open' vs 'Close'):
      Determined by 'Door is opening' vs 'Door is closing' flags and door leaf position trend.
      'Open': position transitions ~0 -> ~700.
      'Close': position transitions ~700 -> ~0.

    - Status ('Normal' vs 'Abnormal resistance'):
      Mechanical resistance (foreign debris, jammed slide rails, deformed seals) forces
      the DC motor to draw elevated current to overcome resistive drag at constant speed.
      The mid-stroke current (middle 50% of the movement duration, 25% - 75%) provides
      clean separation:
        - Open: Normal ~244 mA (<= 253 mA), Abnormal ~431 mA (>= 295 mA). Threshold = 270 mA.
        - Close: Normal ~184 mA (<= 201 mA), Abnormal ~523 mA (>= 311 mA). Threshold = 250 mA.

    Args:
        seg_rows: List of row dictionaries for the segment.
        open_current_threshold: Threshold in mA for Open cycles.
        close_current_threshold: Threshold in mA for Close cycles.

    Returns:
        Dictionary with segment metadata.
    """
    n_rows = len(seg_rows)
    start_time = seg_rows[0]["Datetime"]
    end_time = seg_rows[-1]["Datetime"]

    start_pos = float(seg_rows[0].get("Door leaf position", 0))
    end_pos = float(seg_rows[-1].get("Door leaf position", 0))

    opening_votes = sum(1 for r in seg_rows if r.get("Door is opening") == "1")
    closing_votes = sum(1 for r in seg_rows if r.get("Door is closing") == "1")

    if opening_votes > closing_votes or end_pos > start_pos:
        operation = "Open"
    else:
        operation = "Close"

    # Analyze motor current during mid-stroke (constant speed phase)
    currents = [float(r.get("Motor current(mA)", 0)) for r in seg_rows]
    q1 = int(n_rows * 0.25)
    q3 = int(n_rows * 0.75)
    mid_currents = currents[q1:q3] if q3 > q1 else currents
    mid_stroke_mean_current = sum(mid_currents) / len(mid_currents)

    threshold = (
        open_current_threshold if operation == "Open" else close_current_threshold
    )
    status = "Abnormal resistance" if mid_stroke_mean_current > threshold else "Normal"

    return {
        "start_time": start_time,
        "end_time": end_time,
        "operation": operation,
        "status": status,
        "n_rows": n_rows,
        "mid_stroke_current": round(mid_stroke_mean_current, 2),
    }


def process_door_file(
    input_csv_path: str,
    output_csv_path: Optional[str] = None,
    output_format: str = "answer",
    id_prefix: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Process a continuous door CSV file and return/save identified segments.

    Args:
        input_csv_path: Path to the input continuous door telemetry CSV.
        output_csv_path: Optional path to save the output CSV.
        output_format: 'answer' (Train_Segments_Answer format) or 'prediction' (door_predictions format).
        id_prefix: Prefix for segment_id in 'answer' format (e.g. 'train_seg' or 'test_seg').
                   If None, inferred from input filename.

    Returns:
        List of segment dictionaries.
    """
    with open(input_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(f"Warning: {input_csv_path} is empty.")
        return []

    # Infer id prefix if not supplied
    if id_prefix is None:
        base_name = os.path.basename(input_csv_path).lower()
        if "train" in base_name:
            id_prefix = "train_seg"
        elif "test" in base_name:
            id_prefix = "test_seg"
        else:
            id_prefix = "seg"

    raw_segments = segment_door_data(rows)
    results = []

    for idx, seg_rows in enumerate(raw_segments):
        info = classify_segment(seg_rows)
        seg_id = f"{id_prefix}_{idx + 1:03d}"
        info["segment_id"] = seg_id
        results.append(info)

    if output_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            if output_format == "prediction":
                fieldnames = ["start_time", "end_time", "prediction"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in results:
                    writer.writerow({
                        "start_time": r["start_time"],
                        "end_time": r["end_time"],
                        "prediction": r["status"],
                    })
            else:  # default: 'answer' format
                fieldnames = [
                    "segment_id",
                    "start_time",
                    "end_time",
                    "operation",
                    "status",
                    "n_rows",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in results:
                    writer.writerow({
                        "segment_id": r["segment_id"],
                        "start_time": r["start_time"],
                        "end_time": r["end_time"],
                        "operation": r["operation"],
                        "status": r["status"],
                        "n_rows": r["n_rows"],
                    })
        print(f"Successfully saved {len(results)} segments to {output_csv_path} (format: {output_format})")

    return results


def run_self_verification() -> bool:
    """Verify segmenter against the official Train_Segments_Answer.csv ground truth."""
    train_csv = os.path.join(
        "data", "NebulaX-Hackathon-ProblemStatement", "PS3", "02_Datasets", "Door", "Train.csv"
    )
    answer_csv = os.path.join(
        "data",
        "NebulaX-Hackathon-ProblemStatement",
        "PS3",
        "02_Datasets",
        "Door",
        "Train_Segments_Answer.csv",
    )

    if not os.path.exists(train_csv) or not os.path.exists(answer_csv):
        print("Data files not found for self-verification.")
        return False

    with open(answer_csv, "r", newline="", encoding="utf-8") as f:
        ground_truth = list(csv.DictReader(f))

    predictions = process_door_file(train_csv, id_prefix="train_seg")

    if len(predictions) != len(ground_truth):
        print(f"Segment count mismatch: expected {len(ground_truth)}, got {len(predictions)}")
        return False

    perfect = True
    for idx, (pred, truth) in enumerate(zip(predictions, ground_truth)):
        if (
            pred["segment_id"] != truth["segment_id"]
            or pred["start_time"] != truth["start_time"]
            or pred["end_time"] != truth["end_time"]
            or pred["operation"] != truth["operation"]
            or pred["status"] != truth["status"]
            or pred["n_rows"] != int(truth["n_rows"])
        ):
            print(f"Mismatch at segment {idx + 1}:")
            print(f"  Got: {pred}")
            print(f"  Expected: {truth}")
            perfect = False
            break

    if perfect:
        print(f"Self-verification PASSED: 100% exact match across all {len(ground_truth)} segments and labels!")
    return perfect


def main():
    parser = argparse.ArgumentParser(
        description="Extract and classify door movement segments from continuous telemetry CSV."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Path to input continuous telemetry CSV (e.g. Train.csv or Test.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["answer", "prediction"],
        default="answer",
        help="Output CSV format: 'answer' (segment_id,start_time,end_time,operation,status,n_rows) or 'prediction' (start_time,end_time,prediction)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run self-verification against ground truth Train_Segments_Answer.csv.",
    )

    args = parser.parse_args()

    if args.verify:
        success = run_self_verification()
        sys.exit(0 if success else 1)

    if not args.input:
        parser.print_help()
        sys.exit(1)

    results = process_door_file(
        input_csv_path=args.input,
        output_csv_path=args.output,
        output_format=args.format,
    )
    print(f"Identified {len(results)} segments from {args.input}.")


if __name__ == "__main__":
    main()

