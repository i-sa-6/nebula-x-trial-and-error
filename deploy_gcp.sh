#!/bin/bash
# ==============================================================================
# NebulaX 2026 - Automated Google Cloud Run Deployment Script
# Deploys the Rail Corrugation Digital Twin to Google Cloud
# ==============================================================================
set -e

echo "================================================================="
echo " 🚆 NEBULA-X: GOOGLE CLOUD RUN DEPLOYMENT ENGINE"
echo "================================================================="

# 1. Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI is not installed or not in PATH."
    exit 1
fi

# 2. Check active GCP Project
PROJECT_ID=$(gcloud config get-value project 2>/dev/null || true)
if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" == "(unset)" ]; then
    echo "⚠️ No GCP Project is currently configured in gcloud."
    echo "Please enter your NebulaX GCP Project ID:"
    read -r PROJECT_ID
    gcloud config set project "$PROJECT_ID"
fi

echo "[✓] Active Google Cloud Project: $PROJECT_ID"

# 3. Deploy using Google Cloud Build directly to Cloud Run
echo ""
echo "[*] Submitting build to Google Cloud Build & deploying to Cloud Run (asia-southeast1)..."
echo "    (No local Docker needed — builds entirely on Google Cloud infrastructure)"

gcloud run deploy nebula-rail-twin \
  --source . \
  --region asia-southeast1 \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --project "$PROJECT_ID"

echo ""
echo "================================================================="
echo " 🎉 DEPLOYMENT COMPLETE! YOUR SUBMISSION IS LIVE ON GOOGLE CLOUD!"
echo "================================================================="
SERVICE_URL=$(gcloud run services describe nebula-rail-twin --platform managed --region asia-southeast1 --format 'value(status.url)' 2>/dev/null || true)
if [ -n "$SERVICE_URL" ]; then
    echo " 🌐 Live Public URL: $SERVICE_URL"
    echo " Share this URL in your submission write-up and video presentation."
fi
echo "================================================================="
