import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import shutil

# =========================================================
# CONFIG
# =========================================================
video_folder = Path("assets/Data/Video_V")
output_folder = video_folder / "dataset_3class"
test_videos_folder = video_folder / "test_videos"
SAMPLE_EVERY = 6           # keep 1 of every 15 frames
RANDOM_OFFSET = True        # randomize starting offset per video
BG_CROP_SIZE = 64           # background patch size in pixels

# Explicitly excluded videos (moved to test set)
EXCLUDED_VIDEOS = [
    "V_BIRD_019.mp4",
    "V_BIRD_045.mp4",
    "V_BIRD_046.mp4",
    "V_DRONE_001.mp4",
    "V_DRONE_027.mp4",
    "V_DRONE_048.mp4"
]

classes = ["BACKGROUND", "BIRD", "DRONE"]

# Ensure output directories exist
for cls in classes:
    (output_folder / cls).mkdir(parents=True, exist_ok=True)
for cls in ["BIRD", "DRONE"]:
    (test_videos_folder / cls).mkdir(parents=True, exist_ok=True)

csv_files = sorted(video_folder.glob("*_export.csv"))
if not csv_files:
    raise FileNotFoundError("No *_export.csv files found in", video_folder)


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def copy_specific_test_videos(csv_files, excluded_videos):
    """Copy specific videos to test folder based on the given exclusion list."""
    excluded_csvs = []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if "Source" not in df.columns or df.empty:
            continue
        video_name = df["Source"].iloc[0]
        if video_name in excluded_videos:
            src_vid = video_folder / video_name
            if not src_vid.exists():
                print(f"Warning: {video_name} not found, skipping.")
                continue

            # Determine class folder
            label = "BIRD" if "BIRD" in video_name.upper() else "DRONE"
            dest = test_videos_folder / label
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_vid, dest / src_vid.name)
            print(f"Copied {src_vid.name} → {dest}")
            excluded_csvs.append(csv_path)

    # Save log of excluded videos
    log_path = test_videos_folder / "selected_test_videos.txt"
    with open(log_path, "w") as f:
        f.write("Explicitly excluded test videos:\n\n")
        for name in excluded_videos:
            f.write(f"- {name}\n")
    print(f"\n🧾 Test video list saved to {log_path}")

    # Return CSVs not corresponding to excluded videos
    remaining_csvs = [c for c in csv_files if c not in excluded_csvs]
    return remaining_csvs


def export_from_csv(csv_path: Path):
    """Export cropped Bird/Drone/Background samples from CSV annotations."""
    df = pd.read_csv(csv_path)
    required = {"Source", "Frame", "Label", "X", "Y", "W", "H"}
    if not required.issubset(df.columns):
        print(f"Skipping {csv_path.name} (missing columns)")
        return

    video_name = df["Source"].iloc[0]
    vid_path = video_folder / video_name
    if not vid_path.exists():
        print(f"Video not found: {vid_path}")
        return

    # Random offset for frame sampling
    offset = np.random.randint(0, SAMPLE_EVERY) if RANDOM_OFFSET else 0
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        print(f"Cannot open video: {vid_path}")
        return

    grouped = df.groupby("Frame")
    kept_frames = [f for f in grouped.groups.keys() if ((int(f)-1 - offset) % SAMPLE_EVERY == 0)]
    kept_frames.sort()

    base = csv_path.stem
    saved_counts = {cls: 0 for cls in classes}

    for frame_idx in kept_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx) - 1)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Export Bird/Drone crops
        for _, row in grouped.get_group(frame_idx).iterrows():
            label = str(row["Label"]).upper()
            if label not in ["BIRD", "DRONE"]:
                continue
            x, y, w, h = map(int, [row["X"], row["Y"], row["W"], row["H"]])
            x, y = max(0, x), max(0, y)
            crop = frame[y:y+h, x:x+w]
            if crop.size == 0:
                continue
            out_name = f"{label}_{base}_F{int(frame_idx):06d}.jpg"
            cv2.imwrite(str(output_folder / label / out_name), crop)
            saved_counts[label] += 1
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)

        # Generate one random BACKGROUND crop
        inv_mask = cv2.bitwise_not(mask)
        ys, xs = np.where(inv_mask > 0)
        if len(xs) > 0:
            x0, y0 = np.random.choice(xs), np.random.choice(ys)
            x0 = np.clip(x0, 0, frame.shape[1]-BG_CROP_SIZE)
            y0 = np.clip(y0, 0, frame.shape[0]-BG_CROP_SIZE)
            crop_bg = frame[y0:y0+BG_CROP_SIZE, x0:x0+BG_CROP_SIZE]
            if crop_bg.size > 0:
                out_name = f"BACKGROUND_{base}_F{int(frame_idx):06d}.jpg"
                cv2.imwrite(str(output_folder / "BACKGROUND" / out_name), crop_bg)
                saved_counts["BACKGROUND"] += 1

    cap.release()
    print(f"{csv_path.name}: saved {saved_counts}")


# =========================================================
# MAIN EXECUTION
# =========================================================


train_csvs = copy_specific_test_videos(csv_files, EXCLUDED_VIDEOS)

for csv in tqdm(train_csvs, desc="Extracting crops (1-of-15)"):
    export_from_csv(csv)

print("\n Dataset generation complete.")
print(f"Training dataset: {output_folder.resolve()}")
print(f"Test videos:      {test_videos_folder.resolve()}")
print("Folders:", os.listdir(output_folder))
