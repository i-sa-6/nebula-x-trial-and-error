"""
Launcher for NebulaX Rail Corrugation Condition Monitoring Dashboard.
Usage:
    python app/run_app.py [port]
"""
import os
import sys

# Ensure workspace root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.backend.server import run_server

if __name__ == "__main__":
    env_port = os.environ.get("PORT")
    port = int(env_port) if env_port else (int(sys.argv[1]) if len(sys.argv) > 1 else 8080)
    print("=" * 65)
    print(" 🚆 NEBULA-X: RAIL CORRUGATION CONDITION MONITORING APP")
    print("=" * 65)
    print(f" Starting server on port: {port}")
    print(" Press Ctrl+C to stop the server.")
    print("=" * 65)
    run_server(port)

