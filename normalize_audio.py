import os
import subprocess
import imageio_ffmpeg

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

SRC_DIR = "data"
DST_DIR = "data_normalized"
TARGET_SR = "16000"
TARGET_BITRATE = "64k"

def normalize_file(src_path, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cmd = [
        FFMPEG_EXE, "-y", "-i", src_path,
        "-ar", TARGET_SR, "-ac", "1",
        "-codec:a", "libmp3lame", "-b:a", TARGET_BITRATE,
        dst_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

count = 0
for root, dirs, files in os.walk(SRC_DIR):
    for fname in files:
        if not fname.lower().endswith((".m4a", ".mp3", ".wav")):
            continue
        src_path = os.path.join(root, fname)
        rel_path = os.path.relpath(src_path, SRC_DIR)
        # Every output becomes .mp3 regardless of source format, so real
        # and fake files share an identical final codec/bitrate
        rel_path_mp3 = os.path.splitext(rel_path)[0] + ".mp3"
        dst_path = os.path.join(DST_DIR, rel_path_mp3)
        normalize_file(src_path, dst_path)
        count += 1

print(f"Normalized {count} files into {DST_DIR}/")