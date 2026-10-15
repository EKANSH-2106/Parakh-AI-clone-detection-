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

model = joblib.load("voice_clone_detector.pkl")

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

def predict_file(file_path):
    y, _ = librosa.load(file_path, sr=SR)
    segments = get_segments(y)

    fake_probs = []
    classes = model.classes_

    print(f"\nFile: {file_path}")
    print(f"Total segments analyzed: {len(segments)}")

    for i, seg in enumerate(segments):
        mfcc = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=N_MFCC)
        features = np.mean(mfcc.T, axis=0).reshape(1, -1)
        probs = model.predict_proba(features)[0]
        prob_dict = dict(zip(classes, probs))
        fake_probs.append(prob_dict["fake"])
        print(f"  Segment {i+1}: fake={prob_dict['fake']:.1%}  real={prob_dict['real']:.1%}")

    avg_fake_risk = np.mean(fake_probs)
    verdict = "LIKELY CLONED" if avg_fake_risk >= 0.5 else "LIKELY GENUINE"

    print(f"\n  --> Overall risk score: {avg_fake_risk:.1%} chance of being AI-cloned")
    print(f"  --> Verdict: {verdict}")

if __name__ == "__main__":
    import sys
    predict_file(sys.argv[1])