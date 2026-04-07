import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import threading
from collections import Counter, deque
from pathlib import Path

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SignServe Beta",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = Path("models/best_bigru_attention_aug.keras.zip")
CLASS_NAMES_PATH = Path("models/class_names.json")

SEQUENCE_LENGTH = 30
FEATURE_DIM = 225

PREDICT_EVERY_N_FRAMES = 2
SMOOTHING_WINDOW = 5
MIN_STABLE_COUNT = 3
CONFIDENCE_THRESHOLD = 0.80
NO_SIGN_THRESHOLD = 0.50

MAX_HISTORY = 8
NO_SIGN_LABEL = "No sign recognized"

MIN_HAND_FRAMES_IN_WINDOW = 10
HAND_MOTION_THRESHOLD = 0.012
IDLE_RESET_FRAMES = 12

RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

# ============================================================
# THEME / CSS
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #fff8ef 0%, #fffdf8 100%);
        color: #0f172a;
    }
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.18);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        line-height: 1.2;
    }
    .hero p {
        margin: 0.4rem 0 0 0;
        color: #e2e8f0;
        font-size: 1rem;
    }
    .card {
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(241, 245, 249, 0.95);
        border-radius: 22px;
        padding: 1rem 1.1rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        margin-bottom: 1rem;
    }
    .metric-pill {
        display: inline-block;
        background: #fff7ed;
        color: #c2410c;
        border: 1px solid #fdba74;
        padding: 0.45rem 0.75rem;
        border-radius: 999px;
        font-size: 0.9rem;
        margin-right: 0.5rem;
        margin-top: 0.35rem;
    }
    .small-note {
        color: #475569;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CUSTOM LAYER
# ============================================================
class AttentionPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = tf.keras.layers.Dense(1)

    def call(self, inputs, mask=None):
        scores = self.score_dense(inputs)
        if mask is not None:
            mask = tf.cast(mask, scores.dtype)
            mask = tf.expand_dims(mask, axis=-1)
            scores = scores + (1.0 - mask) * tf.constant(-1e9, dtype=scores.dtype)
        weights = tf.nn.softmax(scores, axis=1)
        context = tf.reduce_sum(weights * inputs, axis=1)
        return context

    def compute_mask(self, inputs, mask=None):
        return None


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource(show_spinner=True)
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"AttentionPooling": AttentionPooling},
        compile=False,
    )
    return model, class_names


MODEL, CLASS_NAMES = load_artifacts()

# ============================================================
# HELPERS
# ============================================================
def extract_landmarks(results):
    pose = np.zeros((33, 3), dtype=np.float32)
    left_hand = np.zeros((21, 3), dtype=np.float32)
    right_hand = np.zeros((21, 3), dtype=np.float32)

    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark[:33]):
            pose[i] = [lm.x, lm.y, lm.z]

    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark[:21]):
            left_hand[i] = [lm.x, lm.y, lm.z]

    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark[:21]):
            right_hand[i] = [lm.x, lm.y, lm.z]

    frame = np.concatenate([
        pose.flatten(),
        left_hand.flatten(),
        right_hand.flatten(),
    ]).astype(np.float32)
    return frame


def normalize_frame(frame_225, eps=1e-6):
    arr = frame_225.reshape(75, 3).copy()

    pose = arr[:33]
    left_hand = arr[33:54]
    right_hand = arr[54:75]

    left_shoulder = pose[11]
    right_shoulder = pose[12]

    left_ok = np.any(np.abs(left_shoulder) > eps)
    right_ok = np.any(np.abs(right_shoulder) > eps)

    if left_ok and right_ok:
        center = (left_shoulder + right_shoulder) / 2.0
        scale = np.linalg.norm(left_shoulder - right_shoulder)
        if scale < eps:
            scale = 1.0

        def norm_block(block):
            out = block.copy()
            mask = np.any(np.abs(out) > eps, axis=1)
            out[mask] = (out[mask] - center) / scale
            return out

        pose = norm_block(pose)
        left_hand = norm_block(left_hand)
        right_hand = norm_block(right_hand)
        arr = np.concatenate([pose, left_hand, right_hand], axis=0)

    return arr.flatten().astype(np.float32)


def hands_visible(results):
    return (
        results.left_hand_landmarks is not None
        or results.right_hand_landmarks is not None
    )


def hand_activity_stats(sequence, eps=1e-6):
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != FEATURE_DIM:
        return 0, 0.0

    seq_3d = seq.reshape(len(seq), 75, 3)
    hands = seq_3d[:, 33:75, :]

    active_mask = np.any(np.abs(hands) > eps, axis=(1, 2))
    active_count = int(np.sum(active_mask))

    if active_count < 2:
        return active_count, 0.0

    active_hands = hands[active_mask]
    diffs = np.abs(np.diff(active_hands, axis=0))
    motion_score = float(np.mean(diffs))
    return active_count, motion_score


def should_run_prediction(sequence_buffer, results):
    if len(sequence_buffer) < SEQUENCE_LENGTH:
        return False, "Warming up"

    if not hands_visible(results):
        return False, "No hands detected"

    active_frames, motion_score = hand_activity_stats(sequence_buffer)

    if active_frames < MIN_HAND_FRAMES_IN_WINDOW:
        return False, "Waiting for hand frames"

    if motion_score < HAND_MOTION_THRESHOLD:
        return False, "No sign motion"

    return True, "Predicting"


def get_stable_prediction(pred_queue):
    if len(pred_queue) == 0:
        return NO_SIGN_LABEL, 0.0, "Idle"

    labels = [x[0] for x in pred_queue]
    counts = Counter(labels)
    best_label, best_count = counts.most_common(1)[0]

    best_confs = [conf for label, conf in pred_queue if label == best_label]
    avg_conf = float(np.mean(best_confs)) if best_confs else 0.0

    if best_count >= MIN_STABLE_COUNT and avg_conf >= CONFIDENCE_THRESHOLD:
        return best_label, avg_conf, "Recognized"

    max_conf = max(conf for _, conf in pred_queue)
    if max_conf < NO_SIGN_THRESHOLD:
        return NO_SIGN_LABEL, max_conf, "Low confidence"

    return NO_SIGN_LABEL, avg_conf, "Unstable"


def draw_landmarks(frame, results, mp_drawing, mp_holistic):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
        )

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )


def draw_right_panel(width, height, display_label, display_conf, status, recent_preds, history):
    panel = np.full((height, width, 3), 247, dtype=np.uint8)

    y = 40
    cv2.putText(panel, "SignServe Beta", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    y += 45

    cv2.rectangle(panel, (20, y), (width - 20, y + 100), (231, 243, 255), -1)
    cv2.rectangle(panel, (20, y), (width - 20, y + 100), (191, 219, 254), 2)
    cv2.putText(panel, "Prediction", (34, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (55, 65, 81), 2)
    cv2.putText(panel, display_label[:26], (34, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (15, 23, 42), 2)
    y += 130

    cv2.putText(panel, f"Confidence: {display_conf:.2f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (51, 65, 85), 2)
    y += 35
    cv2.putText(panel, f"Status: {status}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (51, 65, 85), 2)
    y += 45

    cv2.putText(panel, "Recent predictions", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2)
    y += 30
    if len(recent_preds) == 0:
        cv2.putText(panel, "-", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (100, 100, 100), 2)
        y += 25
    else:
        for label, conf in list(recent_preds)[-5:]:
            cv2.putText(panel, f"{label[:18]} ({conf:.2f})", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (80, 80, 80), 2)
            y += 24

    y += 20
    cv2.putText(panel, "History", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2)
    y += 30
    if len(history) == 0:
        cv2.putText(panel, "-", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (100, 100, 100), 2)
    else:
        for item in history[-MAX_HISTORY:]:
            cv2.putText(panel, item[:24], (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (50, 50, 50), 2)
            y += 26
            if y > height - 20:
                break

    return panel


# ============================================================
# VIDEO PROCESSOR
# ============================================================
class SignVideoProcessor:
    def __init__(self):
        self.model = MODEL
        self.class_names = CLASS_NAMES
        self.lock = threading.Lock()

        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils

        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.prediction_queue = deque(maxlen=SMOOTHING_WINDOW)
        self.history = []

        self.frame_count = 0
        self.idle_frame_counter = 0
        self.display_label = NO_SIGN_LABEL
        self.display_conf = 0.0
        self.status = "Idle"
        self.last_stable_label = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)

        draw_landmarks(img, results, self.mp_drawing, self.mp_holistic)

        raw_feat = extract_landmarks(results)
        norm_feat = normalize_frame(raw_feat)
        self.sequence_buffer.append(norm_feat)

        current_hands_visible = hands_visible(results)
        if current_hands_visible:
            self.idle_frame_counter = 0
        else:
            self.idle_frame_counter += 1

        self.frame_count += 1

        if self.idle_frame_counter >= IDLE_RESET_FRAMES:
            self.sequence_buffer.clear()
            self.prediction_queue.clear()
            self.display_label = NO_SIGN_LABEL
            self.display_conf = 0.0
            self.status = "Idle"
            self.last_stable_label = None

        elif len(self.sequence_buffer) == SEQUENCE_LENGTH and self.frame_count % PREDICT_EVERY_N_FRAMES == 0:
            can_predict, gate_status = should_run_prediction(self.sequence_buffer, results)

            if can_predict:
                seq = np.array(self.sequence_buffer, dtype=np.float32)
                seq = np.expand_dims(seq, axis=0)

                probs = self.model.predict(seq, verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                pred_conf = float(probs[pred_idx])
                pred_label = self.class_names[pred_idx]

                if pred_conf >= CONFIDENCE_THRESHOLD:
                    self.prediction_queue.append((pred_label, pred_conf))
                    self.display_label, self.display_conf, self.status = get_stable_prediction(self.prediction_queue)
                else:
                    self.prediction_queue.clear()
                    self.display_label = NO_SIGN_LABEL
                    self.display_conf = pred_conf
                    self.status = "Low confidence"
                    self.last_stable_label = None
            else:
                self.prediction_queue.clear()
                self.display_label = NO_SIGN_LABEL
                self.display_conf = 0.0
                self.status = gate_status
                self.last_stable_label = None
        else:
            if len(self.sequence_buffer) < SEQUENCE_LENGTH:
                self.display_label = NO_SIGN_LABEL
                self.display_conf = 0.0
                self.status = "Warming up"

        if self.display_label not in [NO_SIGN_LABEL, "Detecting..."]:
            if self.display_label != self.last_stable_label:
                self.history.append(self.display_label)
                if len(self.history) > MAX_HISTORY:
                    self.history = self.history[-MAX_HISTORY:]
                self.last_stable_label = self.display_label
        else:
            self.last_stable_label = None

        right_panel = draw_right_panel(
            width=360,
            height=img.shape[0],
            display_label=self.display_label,
            display_conf=self.display_conf,
            status=self.status,
            recent_preds=self.prediction_queue,
            history=self.history,
        )

        combined = cv2.hconcat([img, right_panel])

        cv2.putText(
            combined,
            "Allow camera access and click START to begin beta test",
            (20, combined.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (245, 158, 11),
            2,
        )

        return av.VideoFrame.from_ndarray(combined, format="bgr24")


# ============================================================
# UI
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>🍽️ SignServe Beta</h1>
        <p>A restaurant-friendly sign-to-text experience with a polished browser UI and live beta testing support.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([2.3, 1.1], gap="large")

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Live recognition")
    st.caption("Click Start, allow camera access, and test the model directly in your browser.")

    webrtc_streamer(
        key="signserve-beta",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=SignVideoProcessor,
        async_processing=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown(
        """
        <div class="card">
            <h3 style="margin-top:0; color:#0f172a;">Beta test checklist</h3>
            <div class="metric-pill">Camera access enabled</div>
            <div class="metric-pill">Good lighting</div>
            <div class="metric-pill">Hands visible</div>
            <div class="metric-pill">Stable internet</div>
            <p class="small-note" style="margin-top:0.9rem;">
                This version is designed for browser testing. The camera feed and predictions are rendered inside the app,
                so testers only need the link and camera permission.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
            <h3 style="margin-top:0; color:#0f172a;">Recommended colors</h3>
            <p class="small-note">
                Primary: <b>#0F172A</b><br>
                Accent: <b>#F59E0B</b><br>
                Background: <b>#FFF8EF</b><br>
                Success: <b>#10B981</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
            <h3 style="margin-top:0; color:#0f172a;">Notes</h3>
            <ul class="small-note">
                <li>If the restaurant Wi-Fi is strict, you may later need a TURN server for more reliable WebRTC connections.</li>
                <li>Keep your model and class_names.json inside the repo under <code>models/</code>.</li>
                <li>For best results, stand where both hands are clearly visible.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()
st.caption("Built with Streamlit, MediaPipe, TensorFlow, and streamlit-webrtc.")
