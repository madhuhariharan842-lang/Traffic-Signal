#!/bin/bash

echo "🚦 AI Traffic Control System - Mac/Linux Setup"
echo "============================================"

echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "Downloading YOLO weights (first time only)..."
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

echo ""
echo "Starting AI Traffic Control System..."
echo "Open your browser to: http://localhost:8501"
echo "Press Ctrl+C to stop the system"
echo ""

streamlit run streamlit_app.py --server.port 8501 --server.address localhost
