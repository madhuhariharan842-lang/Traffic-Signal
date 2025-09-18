"""
Streamlit dashboard to monitor AI traffic controller.
- reads state.json updated by adaptive_signal.py
- shows live camera feed by embedding the ESP32 stream or sample video (browser-side)
- provides manual control buttons that write command.json for adaptive_signal.py to pick up
"""

import streamlit as st
import time
import json
import os
from datetime import datetime

STATE_FILE = "state.json"
COMMAND_FILE = "command.json"
CONFIG_FILE = "config.json"

st.set_page_config(page_title="AI Traffic Signal - Dashboard", layout="wide")

# Load config
with open(CONFIG_FILE, "r") as f:
    cfg = json.load(f)

# Sidebar
st.sidebar.title("Controls")
st.sidebar.markdown("Manual override and settings")

duration = st.sidebar.number_input("Manual override duration (s)", min_value=5, max_value=300, value=15, step=5)

if st.sidebar.button("Set NS Green (Manual)"):
    command = {"cmd":"NS_GREEN", "duration_s": duration}
    with open(COMMAND_FILE, "w") as f:
        json.dump(command, f)
    st.sidebar.success("Command written: NS_GREEN")

if st.sidebar.button("Set EW Green (Manual)"):
    command = {"cmd":"EW_GREEN", "duration_s": duration}
    with open(COMMAND_FILE, "w") as f:
        json.dump(command, f)
    st.sidebar.success("Command written: EW_GREEN")

if st.sidebar.button("Set ALL RED (Manual)"):
    command = {"cmd":"ALL_RED", "duration_s": duration}
    with open(COMMAND_FILE, "w") as f:
        json.dump(command, f)
    st.sidebar.success("Command written: ALL_RED")

st.sidebar.markdown("---")
st.sidebar.write("Video source:")
st.sidebar.write(cfg.get("video_source"))

# Main layout
col1, col2 = st.columns([2,1])

with col1:
    st.header("Live Camera / Video")
    # show embedded ESP32 stream if esp32; else show sample video link placeholder
    if cfg.get("video_source") == "esp32":
        esp_url = cfg.get("esp32_stream_url")
        st.markdown(f"**ESP32 stream:** {esp_url}")
        # embed stream in iframe (works for MJPEG). Some browsers may require additional handling.
        st.image_placeholder = st.empty()
        st.image_placeholder.image("placeholder.png")  # fallback

        # We will show frames by reading state.json images are not stored; alternatively, show iframe:
        st.markdown(f'<iframe src="{esp_url}" width="640" height="480"></iframe>', unsafe_allow_html=True)
    else:
        st.write("Using sample video from disk (preview below)")
        # We can't stream local mp4 directly via Streamlit easily; provide file link and show periodic frame from state
        st.info("If you want to see the frame, run adaptive_signal.py locally and it will write current frames to disk (optional).")

with col2:
    st.header("Signal Status")
    state_box = st.empty()
    counts_box = st.empty()
    ambulance_box = st.empty()
    last_cmd_box = st.empty()

# Live update loop
while True:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
        except Exception:
            state = None
    else:
        state = None

    if state:
        # display signal states
        ns = state["signal_states"]["north_south"]
        ew = state["signal_states"]["east_west"]
        counts = state["counts"]
        ambulance = state.get("ambulance_dirs", [])

        st.experimental_rerun()  # quick UI refresh; streamlit limitations: rerun to reflect updates
    else:
        st.write("Waiting for adaptive_signal.py to produce state.json...")
        time.sleep(1)
        st.experimental_rerun()
