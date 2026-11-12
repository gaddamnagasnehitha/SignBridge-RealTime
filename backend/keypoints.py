"""MediaPipe Holistic keypoint extraction (pose + face + both hands)."""
import cv2
import numpy as np

from . import config


def create_holistic(model_complexity: int = config.MODEL_COMPLEXITY):
    import mediapipe as mp  # imported lazily so training works without a camera stack
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _flatten(landmarks, count: int, with_visibility: bool = False) -> np.ndarray:
    dim = 4 if with_visibility else 3
    if landmarks is None:
        return np.zeros(count * dim, dtype=np.float32)
    if with_visibility:
        arr = [[l.x, l.y, l.z, l.visibility] for l in landmarks.landmark]
    else:
        arr = [[l.x, l.y, l.z] for l in landmarks.landmark]
    return np.asarray(arr, dtype=np.float32).flatten()


def extract_keypoints(results) -> np.ndarray:
    """Return one flat feature vector of length config.FEATURE_DIM."""
    parts = [_flatten(results.pose_landmarks, 33, True)]
    if config.USE_FACE:
        parts.append(_flatten(results.face_landmarks, 468))
    parts.append(_flatten(results.left_hand_landmarks, 21))
    parts.append(_flatten(results.right_hand_landmarks, 21))
    vec = np.concatenate(parts)
    assert vec.shape[0] == config.FEATURE_DIM, "feature size mismatch"
    return vec


def hands_visible(results) -> bool:
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None


def process_bgr_frame(holistic, frame_bgr):
    """Run Holistic on one BGR frame. Returns (feature_vector, hands_detected)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = holistic.process(rgb)
    return extract_keypoints(results), hands_visible(results)
