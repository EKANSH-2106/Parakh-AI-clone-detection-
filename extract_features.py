import os
import imageio_ffmpeg
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

import librosa
import numpy as np
import pandas as pd
DATA_DIR = "data_normalized"
SEGMENT_DURATION = 2.0   # seconds per segment
SEGMENT_OVERLAP = 1.0    # seconds of overlap between consecutive segments
N_MFCC = 40
SR = 22050

def extract_segments(file_path, sr=SR):
    try:
        y, _ = librosa.load(file_path, sr=sr)
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        return []

    segment_len = int(SEGMENT_DURATION * sr)
    hop_len = int((SEGMENT_DURATION - SEGMENT_OVERLAP) * sr)

    segments = []
    start = 0
    while start + segment_len <= len(y):
        segments.append(y[start:start + segment_len])
        start += hop_len

    # If the file is shorter than one segment, pad it so we still get a sample
    if not segments and len(y) > 0:
        padded = np.pad(y, (0, max(0, segment_len - len(y))))
        segments.append(padded[:segment_len])

    return segments

def extract_mfcc_features(segment, sr=SR, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)  # average across time -> fixed-length feature vector

rows = []

for country in os.listdir(DATA_DIR):
    country_path = os.path.join(DATA_DIR, country)
    if not os.path.isdir(country_path):
        continue
    for gender in os.listdir(country_path):
        gender_path = os.path.join(country_path, gender)
        if not os.path.isdir(gender_path):
            continue
        for speaker_id in os.listdir(gender_path):
            speaker_path = os.path.join(gender_path, speaker_id)
            if not os.path.isdir(speaker_path):
                continue
            for fname in os.listdir(speaker_path):
                fpath = os.path.join(speaker_path, fname)
                label = "real" if fname.startswith("original") else "fake"

                for seg in extract_segments(fpath):
                    features = extract_mfcc_features(seg)
                    rows.append(list(features) + [label, country, gender, speaker_id, fname])

feature_cols = [f"mfcc_{i}" for i in range(N_MFCC)]
columns = feature_cols + ["label", "country", "gender", "speaker_id", "source_file"]

df = pd.DataFrame(rows, columns=columns)
df.to_csv("features.csv", index=False)

print("Done. Total segments extracted:", len(df))
print(df["label"].value_counts())