import os
import time
import threading
import cv2
import numpy as np
from flask import Flask, jsonify, request, Response
from flask_cors import CORS

from ac_simulation    import ACSimulation
from inference_engine import predict_frame, set_thresholds as ie_set_thresholds

app = Flask(__name__)
CORS(app)

# ── Shared state ──────────────────────────────────────────────────────────
ac = ACSimulation(room_temp=30.0)
_state = {
    "occupancy":        "LOW",
    "confidence":       0.0,
    "raw_scores":       {},
    "fps":              0.0,
    "frame_count":      0,
    "start_time":       time.time(),
    "video_loaded":     False,
    "video_name":       "",
    "video_progress":   0.0,
    "person_count":     0,
    "janitor_detected": False,
}
_lock = threading.Lock()

# ── FPS Override (set by /api/config) ─────────────────────────────────────
_video_fps_override = None   # None = use native video FPS

# ── Video playback control ─────────────────────────────────────────────────
_video_paused = False

# Latest JPEG frame for MJPEG stream
_latest_frame_jpg = None
_frame_lock       = threading.Lock()

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Occupancy colours for overlay (BGR) ──────────────────────────────────
OCC_COLORS = {
    "low":    (118, 230,   0),   # green
    "medium": (  0, 180, 255),   # amber
    "high":   ( 60,  60, 255),   # red
}

def _draw_overlay(frame, result):
    """Draw student/janitor bounding boxes with color coding and occupancy banner."""
    h, w = frame.shape[:2]
    occ              = result["occupancy"].lower()
    conf             = result["confidence"]
    count            = result.get("person_count", 0)
    janitor_detected = result.get("janitor_detected", False)
    class_counts     = result.get("class_counts", {})

    # Color map per class (BGR)
    CLASS_COLORS = {
        "student": (50, 230, 50),    # bright green
        "janitor": (220, 60, 220),   # purple
    }
    occ_color = OCC_COLORS.get(occ, (200, 200, 200))

    # Draw labeled bounding boxes (student=green, janitor=purple)
    labeled_bboxes = result.get("bboxes_labeled", [])
    if labeled_bboxes:
        for idx, entry in enumerate(labeled_bboxes):
            x, y, bw, bh = entry[0], entry[1], entry[2], entry[3]
            cls_name = entry[4] if len(entry) > 4 else "student"
            score    = entry[5] if len(entry) > 5 else 0.0
            color = CLASS_COLORS.get(cls_name, (200, 200, 200))
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, 2)
            label_txt = f"{cls_name} {score:.0%}" if score > 0 else cls_name
            cv2.putText(frame, label_txt, (x+4, y+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    else:
        # Fallback: plain bboxes list without labels
        for idx, (x, y, bw, bh) in enumerate(result.get("bboxes", []), 1):
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), occ_color, 2)
            cv2.putText(frame, f"#{idx}", (x+4, y+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, occ_color, 2, cv2.LINE_AA)

    # Janitor warning banner (bottom of frame)
    if janitor_detected:
        cv2.rectangle(frame, (0, h-40), (w, h), (60, 20, 160), -1)
        cv2.putText(frame, "JANITOR DETECTED — AC STAYS OFF",
                    (10, h-12), cv2.FONT_HERSHEY_DUPLEX, 0.7, (220, 200, 255), 2, cv2.LINE_AA)

    # Top banner background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 64), (10, 14, 26), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # Occupancy label
    cv2.putText(frame, occ.upper(), (10, 44),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, occ_color, 2, cv2.LINE_AA)

    # Student/janitor count top right
    n_students = class_counts.get("student", count)
    n_janitors = class_counts.get("janitor", 0)
    if n_janitors > 0:
        cnt_txt = f"{n_students} student{'s' if n_students!=1 else ''} | {n_janitors} janitor"
    elif n_students > 0:
        cnt_txt = f"{n_students} student{'s' if n_students!=1 else ''}"
    else:
        cnt_txt = f"AI: {int(conf*100)}%"
    cv2.putText(frame, cnt_txt, (w - 370, 44),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, occ_color, 2, cv2.LINE_AA)

    # Confidence bar
    bar_x = 210; bar_y = 22; bar_h = 16
    bar_w = int((w - bar_x - 230) * conf)
    cv2.rectangle(frame, (bar_x, bar_y), (w - 230, bar_y + bar_h), (40, 40, 40), -1)
    if bar_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), occ_color, -1)

    return frame




def _make_blank_frame():
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (13, 20, 40)
    cv2.putText(img, "Upload a video to begin", (90, 175),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (100, 150, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "Use the Upload Video button above", (80, 215),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 90, 130), 1, cv2.LINE_AA)
    _, jpg = cv2.imencode(".jpg", img)
    return jpg.tobytes()


# ── Asynchronous AI Inference & Video Playback ──────────────────────────────
_current_raw_frame = None
_latest_result = {
    "occupancy": "low",
    "confidence": 0.95,
    "raw_scores": {},
    "person_count": 0,
    "bboxes": []
}
_result_lock = threading.Lock()
_video_active = False
# Event fired every time a new JPEG frame is ready — MJPEG generator waits on this
_frame_ready  = threading.Event()

def _bg_inference_worker():
    """Runs AI inference on the latest frame in the background at regular intervals without slowing down video playback."""
    global _latest_result
    while True:
        try:
            with _result_lock:
                frame_to_process = _current_raw_frame.copy() if _current_raw_frame is not None else None
            
            if frame_to_process is not None:
                res = predict_frame(frame_to_process)
                if res:
                    with _result_lock:
                        _latest_result = res
            time.sleep(0.3)  # AI runs ~3 times per second, perfectly smooth for occupancy tracking
        except Exception as e:
            print(f"[server] Async AI inference error: {e}")
            time.sleep(0.5)

threading.Thread(target=_bg_inference_worker, daemon=True).start()


def _video_inference_loop(video_path):
    global _latest_frame_jpg, _current_raw_frame, _video_active

    _video_active = True
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[server] Cannot open: {video_path}")
        return

    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if not native_fps or native_fps <= 0 or native_fps > 240:
        native_fps = 30.0

    print(f"[server] Video native FPS: {native_fps:.2f}  Total frames: {total}")

    fps_counter, fps_timer = 0, time.time()

    # ── PTS anchor — maps video presentation time to wall-clock time ──────────
    # We read the first frame's PTS BEFORE reading to anchor wall-clock to video time
    video_start_wall = time.time()
    video_start_pts  = 0.0   # ms, set after first successful read

    while _video_active:
        try:
            # ── Pause support ─────────────────────────────────────────────
            if _video_paused:
                time.sleep(0.1)
                # Re-anchor after unpause so no time-burst
                video_start_wall = time.time() - (pts_before_ms - video_start_pts) / 1000.0 if video_start_pts > 0 else time.time()
                continue
            pts_before_ms = cap.get(cv2.CAP_PROP_POS_MSEC)   # PTS of THIS frame
            ret, frame    = cap.read()
            if not ret:
                # Loop video back to start — reset PTS anchor
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                video_start_wall = time.time()
                video_start_pts  = 0.0
                time.sleep(0.05)
                continue

            idx      = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            progress = idx / total if total > 0 else 0

            # Anchor: on first frame set the reference point
            if video_start_pts == 0.0 and pts_before_ms >= 0:
                video_start_pts  = pts_before_ms
                video_start_wall = time.time()

            # ── Cap resolution for fast encoding (phones often shoot 4K) ──────
            MAX_WIDTH = 1280
            h, w = frame.shape[:2]
            if w > MAX_WIDTH:
                scale = MAX_WIDTH / w
                frame = cv2.resize(frame, (MAX_WIDTH, int(h * scale)),
                                   interpolation=cv2.INTER_LINEAR)

            # Share raw frame with the async AI thread
            with _result_lock:
                _current_raw_frame = frame
                result = dict(_latest_result)

            # Draw AI overlay (lightweight — bboxes already computed by bg thread)
            frame_overlay = _draw_overlay(frame.copy(), result)
            _, jpg = cv2.imencode(".jpg", frame_overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
            with _frame_lock:
                _latest_frame_jpg = jpg.tobytes()
            _frame_ready.set()   # Signal the MJPEG generator that a new frame is ready

            fps_counter += 1
            elapsed_fps = time.time() - fps_timer
            if elapsed_fps >= 1.0:
                measured_fps = fps_counter / elapsed_fps
                fps_counter  = 0
                fps_timer    = time.time()
            else:
                measured_fps = _state["fps"]

            # Only count people when janitor is NOT active
            janitor_is_active = time.time() < ac.janitor_until
            display_person_count = 0 if janitor_is_active else result.get("person_count", 0)

            with _lock:
                ac.update_janitor_status(result.get("janitor_detected", False))
                ac.update_occupancy(result["occupancy"])
                ac.tick()
                _state.update({
                    "occupancy":        result["occupancy"].upper(),
                    "confidence":       result["confidence"],
                    "raw_scores":       result["raw_scores"],
                    "fps":              round(measured_fps, 1),
                    "frame_count":      _state["frame_count"] + 1,
                    "video_progress":   round(progress, 3),
                    "person_count":     display_person_count,
                    "janitor_detected": result.get("janitor_detected", False),
                })

            # ── PTS-based precise frame pacing ────────────────────────────────
            # Calculate when THIS frame should have been shown on wall-clock
            if _video_fps_override:
                # User-specified FPS override: use fixed intervals from start
                frame_num    = idx - 1
                target_wall  = video_start_wall + frame_num / _video_fps_override
            elif pts_before_ms > 0:
                # VFR-safe: use actual PTS timestamp embedded in the video file
                target_wall  = video_start_wall + (pts_before_ms - video_start_pts) / 1000.0
            else:
                # Fallback: nominal FPS
                frame_num    = idx - 1
                target_wall  = video_start_wall + frame_num / native_fps

            now        = time.time()
            sleep_time = target_wall - now
            if sleep_time > 0.001:
                time.sleep(sleep_time)
            elif sleep_time < -1.0:
                # More than 1s behind — re-anchor to avoid runaway playback
                print(f"[server] Video fell behind by {-sleep_time:.2f}s, re-anchoring")
                video_start_wall = time.time() - pts_before_ms / 1000.0 if pts_before_ms > 0 else time.time()
                video_start_pts  = 0.0

        except Exception as exc:
            print(f"[server] Video loop error (frame {_state.get('frame_count',0)}): {exc}")
            time.sleep(0.05)

    cap.release()


def _start_video(path):
    global _video_active
    _video_active = False
    time.sleep(0.1)
    t = threading.Thread(target=_video_inference_loop, args=(path,), daemon=True)
    t.start()


def _ac_tick_loop():
    while True:
        with _lock:
            ac.tick()
        time.sleep(1)

threading.Thread(target=_ac_tick_loop, daemon=True).start()


@app.route("/api/upload", methods=["POST"])
def api_upload():
    global _video_paused
    if "video" not in request.files:
        return jsonify({"error": "No file sent"}), 400
    f    = request.files["video"]
    name = f.filename or "upload.mp4"
    path = os.path.join(UPLOAD_DIR, name)
    f.save(path)
    # Reset everything to defaults on new video upload
    _video_paused = False
    with _lock:
        # Reset AC to factory defaults
        ac.__init__(room_temp=30.0)
        _state["video_loaded"]  = True
        _state["video_name"]    = name
        _state["frame_count"]   = 0
        _state["start_time"]    = time.time()
        _state["person_count"]  = 0
        _state["janitor_detected"] = False
    _start_video(path)
    return jsonify({"ok": True, "filename": name})


@app.route("/api/video_feed")
def api_video_feed():
    def generate():
        blank = _make_blank_frame()
        last_sent = None
        while True:
            # Wait up to 1s for a new frame — immediately wakes when _frame_ready.set() is called
            _frame_ready.wait(timeout=1.0)
            _frame_ready.clear()
            with _frame_lock:
                jpg = _latest_frame_jpg
            if jpg is None:
                jpg = blank
            # Only send if the frame is actually new (avoid duplicate frames)
            if jpg is not last_sent:
                last_sent = jpg
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    with _lock:
        data = {**_state, **ac.state_dict()}
        data["uptime_secs"] = round(time.time() - _state["start_time"], 1)
    return jsonify(data)


@app.route("/api/config", methods=["POST"])
def api_config():
    global _video_fps_override
    body = request.get_json(force=True)

    # Override occupancy manually
    if "override_occupancy" in body:
        with _lock:
            ac.update_occupancy(body["override_occupancy"])

    # FPS override (None = use native video FPS)
    if "video_fps" in body:
        val = body["video_fps"]
        _video_fps_override = max(1.0, min(120.0, float(val))) if val else None
        print(f"[server] FPS override set to: {_video_fps_override}")

    # Target temperature manual adjustment
    if "target_temp" in body:
        with _lock:
            ac.set_target_temp(int(body["target_temp"]))

    # Threshold updates
    if "thresholds" in body:
        t = body["thresholds"]
        with _lock:
            ac.set_thresholds(
                medium_min  = t.get('medium_min'),
                high_min    = t.get('high_min'),
                medium_temp = t.get('medium_temp'),
                high_temp   = t.get('high_temp'),
            )
            # Also update inference engine so low/medium/high classification uses new values
            ie_set_thresholds(
                medium_min = t.get('medium_min'),
                high_min   = t.get('high_min'),
            )
            # Force re-evaluation so new thresholds take effect immediately
            current_occ = ac.occupancy
            ac._pending_occ   = ""   # break stability lock
            ac._pending_since = 0.0
            ac.update_occupancy(current_occ)

    return jsonify({"ok": True})


@app.route("/api/play", methods=["POST"])
def api_play():
    global _video_paused
    _video_paused = False
    return jsonify({"ok": True, "paused": False})


@app.route("/api/pause", methods=["POST"])
def api_pause():
    global _video_paused
    _video_paused = True
    return jsonify({"ok": True, "paused": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    global _video_active, _video_paused, _latest_frame_jpg, _latest_result
    _video_paused = False
    _video_active = False
    with _frame_lock:
        _latest_frame_jpg = None
    with _result_lock:
        _latest_result = {
            "occupancy": "low",
            "confidence": 0.0,
            "raw_scores": {},
            "person_count": 0,
            "bboxes": []
        }
    with _lock:
        ac.reset()
        _state["video_loaded"]   = False
        _state["video_name"]     = ""
        _state["frame_count"]    = 0
        _state["person_count"]   = 0
        _state["video_progress"] = 0.0
        _state["occupancy"]      = "LOW"
        _state["confidence"]     = 0.0
        _state["raw_scores"]     = {}
        _state["janitor_detected"] = False
    return jsonify({"ok": True})


@app.route("/")
def index():
    dashboard = os.path.join(os.path.dirname(__file__),
                             "..", "dashboard", "index.html")
    with open(dashboard, encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
