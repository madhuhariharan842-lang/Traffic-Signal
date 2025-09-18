"""
adaptive_signal.py
- Reads config.json
- Opens video (sample mp4 or ESP32 stream)
- Runs object detection (YOLOv8 if installed or simple contour fallback)
- Counts vehicles per direction, detects ambulance class if model supports it
- AdaptiveSignalController decides which direction gets green
- Sends serial commands to Arduino (optional)
- Publishes current state to state.json for the dashboard
- Listens to command.json for manual override commands from the UI
"""

import time
import json
import threading
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict
import cv2
import numpy as np

# Try to import ultralytics YOLO (fast detection). If not installed, we'll use a fallback.
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

try:
    import serial
    SERIAL_AVAILABLE = True
except Exception:
    SERIAL_AVAILABLE = False

STATE_FILE = "state.json"
COMMAND_FILE = "command.json"

# -------------------------------
# Configuration loader
# -------------------------------
def load_config(path="config.json"):
    if not os.path.exists(path):
        raise FileNotFoundError("config.json not found. Create or copy the provided config.json")
    with open(path, "r") as f:
        return json.load(f)

# -------------------------------
# Data classes
# -------------------------------
@dataclass
class SignalState:
    current_signal: str  # 'red','yellow','green'
    timer: int
    direction: str       # 'north_south' or 'east_west'
    last_change: str     # ISO timestamp

# -------------------------------
# Simple Arduino controller
# -------------------------------
class ArduinoController:
    def __init__(self, port=None, baud=115200):
        self.port = port
        self.baud = baud
        self.ser = None
        if port and SERIAL_AVAILABLE:
            try:
                self.ser = serial.Serial(port, baud, timeout=1)
                time.sleep(2)
                print(f"[Arduino] Connected to {port}")
            except Exception as e:
                print(f"[Arduino] Serial open failed: {e}")
                self.ser = None
        else:
            if port:
                print("[Arduino] pyserial not installed or available; serial disabled.")

    def send(self, cmd: str):
        if self.ser:
            try:
                self.ser.write((cmd + "\n").encode())
            except Exception as e:
                print(f"[Arduino] Write failed: {e}")
        else:
            # For debug, print the command instead of sending
            print(f"[Arduino] (simulated) -> {cmd}")

    def close(self):
        if self.ser:
            self.ser.close()

# -------------------------------
# Detector class (YOLO or fallback)
# -------------------------------
class Detector:
    def __init__(self, config):
        self.model = None
        self.names = {}
        self.ambulance_class_name = config.get("ambulance_class_name", "ambulance")
        if YOLO_AVAILABLE and config.get("yolo_model", None):
            try:
                self.model = YOLO(config["yolo_model"])
                self.names = self.model.names
                print("[Detector] YOLO model loaded:", config["yolo_model"])
            except Exception as e:
                print("[Detector] YOLO load failed:", e)
                self.model = None

    def predict(self, frame):
        """
        Returns list of detections: [{"label":str, "conf":float, "xyxy":[x1,y1,x2,y2]}, ...]
        """
        if self.model:
            results = self.model(frame, imgsz=640, conf=0.35)
            dets = []
            for r in results:
                # r.boxes -> xyxy, conf, cls
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy().tolist())
                    cls = int(box.cls[0].cpu().numpy().tolist())
                    label = self.names.get(cls, str(cls))
                    dets.append({"label": label, "conf": conf, "xyxy": xyxy})
            return dets
        else:
            # Fallback: detect moving blobs (very rough)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5,5), 0)
            # background subtractor could be used; but quick simple threshold
            _, th = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            dets = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500:  # tune this threshold
                    x,y,w,h = cv2.boundingRect(cnt)
                    dets.append({"label": "vehicle", "conf": 0.5, "xyxy": [x,y,x+w,y+h]})
            return dets

# -------------------------------
# Adaptive controller
# -------------------------------
class AdaptiveSignalController:
    def __init__(self, config):
        self.config = config
        # initial states
        now = datetime.now().isoformat()
        self.signals = {
            "north_south": SignalState("green", config.get("min_green_time", 15), "north_south", now),
            "east_west": SignalState("red", config.get("min_green_time", 15), "east_west", now)
        }
        self.min_green = config.get("min_green_time", 15)
        self.max_green = config.get("max_green_time", 60)
        self.yellow_time = config.get("yellow_time", 3)
        self.emergency_active = False
        self.manual_override = None  # None or dict {"cmd": "NS_GREEN", "until": iso}

    def decide(self, counts: Dict[str,int], ambulance_dirs=set()):
        """
        Determine which major direction should be green.
        counts is dict: north,south,east,west vehicle counts
        ambulance_dirs: set of directions where ambulance detected
        returns command string e.g. "NS_GREEN" or "EW_GREEN" or "ALL_RED"
        """
        # Emergency override: if ambulance in NS -> NS green, if in EW -> EW green
        if len(ambulance_dirs) > 0 and self.config.get("emergency_priority", True):
            # if ambulance in north or south -> north_south
            if any(d in ("north","south") for d in ambulance_dirs):
                return "NS_GREEN"
            else:
                return "EW_GREEN"

        # Manual override check
        if self.manual_override:
            # if still valid
            until = datetime.fromisoformat(self.manual_override["until"])
            if datetime.now() < until:
                return self.manual_override["cmd"]
            else:
                self.manual_override = None  # expire

        # Normal adaptive density-based decision
        ns = counts.get("north",0) + counts.get("south",0)
        ew = counts.get("east",0) + counts.get("west",0)

        # If both zero -> keep current
        if ns==0 and ew==0:
            # keep existing green if any
            for k,s in self.signals.items():
                if s.current_signal == "green":
                    return "NS_GREEN" if k=="north_south" else "EW_GREEN"
            return "NS_GREEN"

        # Choose the side with more vehicles
        if ns >= ew:
            return "NS_GREEN"
        else:
            return "EW_GREEN"

    def apply_command(self, cmd: str):
        """
        Update states when a new command is applied.
        Accepts: NS_GREEN, EW_GREEN, ALL_RED
        """
        now = datetime.now().isoformat()
        if cmd == "NS_GREEN":
            self.signals["north_south"].current_signal = "green"
            self.signals["north_south"].timer = min(self.max_green, max(self.min_green, self.signals["north_south"].timer))
            self.signals["north_south"].last_change = now
            self.signals["east_west"].current_signal = "red"
            self.signals["east_west"].timer = self.min_green
            self.signals["east_west"].last_change = now
        elif cmd == "EW_GREEN":
            self.signals["east_west"].current_signal = "green"
            self.signals["east_west"].timer = min(self.max_green, max(self.min_green, self.signals["east_west"].timer))
            self.signals["east_west"].last_change = now
            self.signals["north_south"].current_signal = "red"
            self.signals["north_south"].timer = self.min_green
            self.signals["north_south"].last_change = now
        elif cmd == "ALL_RED":
            self.signals["east_west"].current_signal = "red"
            self.signals["north_south"].current_signal = "red"
            self.signals["east_west"].last_change = now
            self.signals["north_south"].last_change = now

# -------------------------------
# Utility functions for state/command files
# -------------------------------
def write_state(state: Dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)

def read_command():
    if not os.path.exists(COMMAND_FILE):
        return None
    try:
        with open(COMMAND_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None

# -------------------------------
# Video input helper
# -------------------------------
def open_video_source(config):
    vs = config.get("video_source", "sample")
    if vs == "sample":
        path = config.get("sample_video_path", "traffic_sample.mp4")
        cap = cv2.VideoCapture(path)
    else:
        url = config.get("esp32_stream_url", "")
        cap = cv2.VideoCapture(url)
    return cap

# -------------------------------
# Region mapping - divides frame into four quadrants
# -------------------------------
def map_detection_to_direction(xyxy, frame_w, frame_h):
    x1,y1,x2,y2 = xyxy
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    # split frame vertically and horizontally (simple)
    if cy < frame_h/2 and cx < frame_w/2:
        return "north"
    if cy >= frame_h/2 and cx < frame_w/2:
        return "south"
    if cx >= frame_w/2 and cy < frame_h/2:
        return "east"
    return "west"

# -------------------------------
# Main processing loop
# -------------------------------
def processing_loop(config_path="config.json"):
    config = load_config(config_path)
    cap = open_video_source(config)
    det = Detector(config)
    arduino = ArduinoController(port=config.get("arduino_port"))
    controller = AdaptiveSignalController(config)

    fps_wait = 0.03
    last_state_write = 0

    while True:
        # Check for manual command from UI
        cmd_json = read_command()
        if cmd_json:
            cmd = cmd_json.get("cmd")
            duration = cmd_json.get("duration_s", 10)
            if cmd in ("NS_GREEN","EW_GREEN","ALL_RED"):
                # apply manual override for duration
                until = (datetime.now() + timedelta(seconds=duration)).isoformat()
                controller.manual_override = {"cmd": cmd, "until": until}
                print("[Main] Manual override:", controller.manual_override)
            # remove command file so it doesn't repeat
            try:
                os.remove(COMMAND_FILE)
            except:
                pass

        ret, frame = cap.read()
        if not ret:
            # if using sample video and EOF -> loop again or break
            if config.get("video_source") == "sample":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("[Main] No frame from stream, retrying...")
                time.sleep(1)
                continue

        frame_h, frame_w = frame.shape[:2]
        dets = det.predict(frame)

        # Count vehicles, detect ambulance
        counts = {"north":0,"south":0,"east":0,"west":0}
        ambulance_dirs = set()
        drawn = frame.copy()
        for d in dets:
            label = d["label"].lower()
            x1,y1,x2,y2 = map(int, d["xyxy"])
            dirn = map_detection_to_direction((x1,y1,x2,y2), frame_w, frame_h)
            if label in ("car","truck","bus","motorcycle","vehicle","bicycle"):
                counts[dirn] += 1
            if label == det.ambulance_class_name or "ambulance" in label:
                ambulance_dirs.add(dirn)
                # increase priority count as well
                counts[dirn] += 1

            # draw box & label
            cv2.rectangle(drawn, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(drawn, label, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

        # Decide command from controller
        cmd = controller.decide(counts, ambulance_dirs)
        controller.apply_command(cmd)

        # Send to Arduino
        arduino.send(cmd)

        # Prepare state
        state = {
            "timestamp": datetime.now().isoformat(),
            "counts": counts,
            "ambulance_dirs": list(ambulance_dirs),
            "signal_states": {
                "north_south": vars(controller.signals["north_south"]),
                "east_west": vars(controller.signals["east_west"])
            },
            "last_command": cmd
        }

        # Write state occasionally (not every frame to reduce I/O)
        if time.time() - last_state_write > 0.5:
            write_state(state)
            last_state_write = time.time()

        # small sleep to control CPU usage
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(fps_wait)

    # cleanup
    cap.release()
    arduino.close()

# Allow running as script
if __name__ == "__main__":
    processing_loop("config.json")
