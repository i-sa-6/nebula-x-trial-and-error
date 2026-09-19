"""
Gemini Streaming Engine for NebulaX Rail Condition Monitoring.
Demonstrates token-by-token streaming using `generateContentStream` to avoid UI hanging.
"""
import os
import json
from typing import Generator

def stream_gemini_advisory(prediction: str, confidence: float, speed_kmh: float,
                           wavelength_mm: float, peak_freq_hz: float,
                           urgency: str) -> Generator[str, None, None]:
    """
    Calls Google GenAI `generate_content_stream` to stream engineering advice token-by-token.
    Falls back to deterministic LTA domain engineering advice if no API key is present.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    
    prompt = f"""
You are an expert LTA Permanent Way (P-Way) Railway Track Maintenance Engineer.
A train condition monitoring system detected the following condition:
- Defect Classification: {prediction}
- Model Confidence: {confidence}%
- Train Speed: {speed_kmh} km/h
- Dominant Vibration Peak: {peak_freq_hz} Hz
- Estimated Corrugation Wavelength (lambda = v / f): {wavelength_mm} mm
- Urgency Level: {urgency}

Provide a concise, professional, 2-to-3 sentence maintenance instruction for the track division.
Specify the exact track rail side, grinding/milling speed recommendations, and acoustic verification.
"""

    if not api_key:
        # High-performance built-in domain engineering generator (token streaming)
        if prediction == "Normal":
            base_text = "Both rails operating within healthy acoustic and vibration bounds. No immediate track intervention required. Continue standard scheduled ultrasonic and geometric track inspections."
        elif prediction == "Side I":
            base_text = f"Side I (Left) rail shows characteristic periodic corrugation (est. wavelength ~{wavelength_mm} mm, peak {peak_freq_hz} Hz). Schedule targeted rail milling/grinding possession for Side I track section during the next engineering hours window."
        else:
            base_text = f"Side II (Right) rail shows abnormal corrugation wear (est. wavelength ~{wavelength_mm} mm, peak {peak_freq_hz} Hz). Schedule targeted rail milling/grinding possession for Side II track section to prevent premature bearing and wheelset degradation."

        # Yield tokens with words
        words = base_text.split(" ")
        for i, word in enumerate(words):
            yield word if i == 0 else " " + word
        return

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"Fallback advisory: {e}"
