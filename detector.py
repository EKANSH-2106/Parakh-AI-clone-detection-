import os
import imageio_ffmpeg
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

import librosa
import numpy as np
import joblib

N_MFCC = 40
SR = 22050
SEGMENT_DURATION = 2.0
SEGMENT_OVERLAP = 1.0

_model = None

def get_model():
    global _model
    if _model is None:
        _model = joblib.load("voice_clone_detector.pkl")
    return _model

def get_segments(y, sr=SR):
    segment_len = int(SEGMENT_DURATION * sr)
    hop_len = int((SEGMENT_DURATION - SEGMENT_OVERLAP) * sr)
    segments = []
    start = 0
    while start + segment_len <= len(y):
        segments.append(y[start:start + segment_len])
        start += hop_len
    if not segments and len(y) > 0:
        padded = np.pad(y, (0, max(0, segment_len - len(y))))
        segments.append(padded[:segment_len])
    return segments

def analyze_audio(file_path):
    model = get_model()
    y, _ = librosa.load(file_path, sr=SR)
    segments = get_segments(y)
    classes = model.classes_
    fake_probs = []

    for seg in segments:
        mfcc = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=N_MFCC)
        features = np.mean(mfcc.T, axis=0).reshape(1, -1)
        probs = model.predict_proba(features)[0]
        fake_probs.append(dict(zip(classes, probs))["fake"])

    avg_fake_risk = float(np.mean(fake_probs))
    return {
        "risk_score": avg_fake_risk,
        "verdict": "LIKELY CLONED" if avg_fake_risk >= 0.5 else "LIKELY GENUINE",
        "segment_scores": fake_probs,
    }