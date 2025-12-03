import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import uuid
from collections import deque, Counter

# ============================================================
# Load MobileNetV2 model (trained for 3 classes)
# ============================================================
def load_classifier(model_path="best_mobilenetv2_final.pth"):
    num_classes = 3
    classes = ["BACKGROUND", "BIRD", "DRONE"]

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    print("classifier loaded.")
    return model, preprocess, classes


MODEL, PREPROCESS, CLASS_NAMES = load_classifier()

# ============================================================
# Utility: ROI + Image helpers
# ============================================================
def make_square_roi(frame, x, y, w, h, pad=0.2):
    """Crop a square ROI centered on the bounding box."""
    h_img, w_img = frame.shape[:2]
    side = int(max(w, h) * (1 + pad))
    cx, cy = x + w // 2, y + h // 2
    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(w_img, x0 + side)
    y1 = min(h_img, y0 + side)
    roi = frame[y0:y1, x0:x1]
    return roi, (cx, cy), (x0, y0, x1, y1)


def classify_roi(roi_bgr):
    """Return (label, confidence) from MobileNetV2 classifier."""
    if roi_bgr.size == 0:
        return "BACKGROUND", 0.0
    img_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    tensor = PREPROCESS(img_pil).unsqueeze(0)
    with torch.no_grad():
        out = MODEL(tensor)
        probs = F.softmax(out, dim=1)[0].cpu().numpy()
        conf = float(np.max(probs))
        label = CLASS_NAMES[int(np.argmax(probs))]
    return label, conf


# ============================================================
# Background subtraction
# ============================================================
def compute_two_point_mask(buffer, k1=1, k2=4, threshold_val=10):
    diff1 = cv2.absdiff(buffer[-1], buffer[-1 - k1])
    diff2 = cv2.absdiff(buffer[-1], buffer[-1 - k2])
    fgMask = cv2.bitwise_and(diff1, diff2)
    _, fgMask = cv2.threshold(fgMask, threshold_val, 255, cv2.THRESH_BINARY)
    fgMask = cv2.medianBlur(fgMask, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgMask = cv2.dilate(fgMask, kernel, iterations=15)
    return fgMask


# ============================================================
# Color selection 
# ============================================================
def get_label_color(label: str):
    if label == "DRONE":
        return (0, 255, 0)     # Green
    elif label == "BIRD":
        return (0, 0, 255)     # Red
    else:
        return (0, 255, 255)   # Yellow


# ============================================================
# IoU (Intersection over Union)
# ============================================================
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter_area / float(boxA_area + boxB_area - inter_area)


# ============================================================
# Track class (simplified: no velocity, no smoothing)
# ============================================================
class Track:
    def __init__(self, cx, cy, roi, label, conf, box, timeout_max=20, hist_len=5, avg_classes=True):
        self.id = str(uuid.uuid4())[:8]
        self.cx = cx
        self.cy = cy
        self.box = box
        self.label = label
        self.conf = conf
        self.timeout = 0
        self.age = 0
        self.timeout_max = timeout_max
        self.avg_classes = avg_classes
        self.class_hist = deque(maxlen=hist_len)
        self.conf_hist = deque(maxlen=hist_len)

    def update(self, cx, cy, label, conf, box):
        self.age += 1
        self.cx, self.cy, self.box = cx, cy, box

        if self.avg_classes:
            self.class_hist.append(label)
            self.conf_hist.append(conf)
            label = Counter(self.class_hist).most_common(1)[0][0]
            conf = np.mean(self.conf_hist)

        self.label, self.conf = label, conf
        self.timeout = 0

    def predict(self):
        self.age += 1
        self.timeout += 1


# ============================================================
# MONO-ROI mode
# ============================================================
def process_frame_mono(frame, gray, buffer, state, params, avg_classes=True):
    fgMask = compute_two_point_mask(buffer, params["k1"], params["k2"], params["threshold"])
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    last_bbox = state["last_bbox"]

    best_c = None
    best_dist = float("inf")
    for c in contours:
        if cv2.contourArea(c) < params["min_area"]:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w / 2, y + h / 2
        dist = np.hypot(cx - (last_bbox[0] if last_bbox else cx),
                        cy - (last_bbox[1] if last_bbox else cy))
        if dist < best_dist:
            best_dist, best_c = dist, (c, x, y, w, h)

    if best_c is not None:
        c, x, y, w, h = best_c
        cx, cy = x + w / 2, y + h / 2
        roi, _, box = make_square_roi(frame, x, y, w, h, pad=0.2)
        label, conf = classify_roi(roi)

        if avg_classes:
            state["class_hist"].append(label)
            state["conf_hist"].append(conf)
            label = Counter(state["class_hist"]).most_common(1)[0][0]
            conf = np.mean(state["conf_hist"])

        color = get_label_color(label)
        x0, y0, x1, y1 = map(int, box)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.putText(frame, mode, (50,  50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)
        last_bbox = (x, y, w, h)

    state.update(dict(last_bbox=last_bbox))
    return frame, fgMask, state


# ============================================================
# MULTI-ROI Mode (with overlap removal)
# ============================================================
def process_frame_multi(frame, gray, buffer, tracks, params, avg_classes=True):
    fgMask = compute_two_point_mask(buffer, params["k1"], params["k2"], params["threshold"])
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for c in contours:
        if cv2.contourArea(c) < params["min_area"]:
            continue
        x, y, w, h = cv2.boundingRect(c)
        roi, (cx, cy), box = make_square_roi(frame, x, y, w, h, pad=0.25)
        detections.append((cx, cy, roi, box))

    used = set()
    for track in tracks:
        best_idx, best_dist = -1, float("inf")
        for i, (cx, cy, roi, box) in enumerate(detections):
            if i in used:
                continue
            dist = np.hypot(cx - track.cx, cy - track.cy)
            if dist < best_dist:
                best_dist, best_idx = dist, i
        if best_idx != -1 and best_dist < params["max_match_dist"]:
            used.add(best_idx)
            cx, cy, roi, box = detections[best_idx]
            label, conf = classify_roi(roi)
            track.update(cx, cy, label, conf, box)
        else:
            track.predict()

    new_tracks = []
    for i, (cx, cy, roi, box) in enumerate(detections):
        if i in used:
            continue

        label, conf = classify_roi(roi)
        new_track = Track(cx, cy, roi, label, conf, box,
                          timeout_max=params["timeout_max"], avg_classes=avg_classes)

        overlapping_old = []
        for t in tracks:
            iou = compute_iou(box, t.box)
            if iou >= params["overlap_thresh"]:
                overlapping_old.append(t)

        if overlapping_old:
            oldest = max(overlapping_old, key=lambda t: t.age)
            if oldest in tracks:
                tracks.remove(oldest)
               

        new_tracks.append(new_track)

    tracks.extend(new_tracks)
    tracks[:] = [t for t in tracks if t.timeout < t.timeout_max]

    for track in tracks:
        color = get_label_color(track.label)
        x0, y0, x1, y1 = map(int, track.box)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        cv2.putText(frame, f"{track.label} ({track.conf:.2f})",
                    (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, (int(track.cx), int(track.cy)), 4, color, -1)

    return frame, fgMask, tracks


# ============================================================
# Main loop
# ============================================================
def run_video_pipeline(video_path, params, mode="mono", avg_classes=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video.")
        return

    buffer = []
    if mode == "multi":
        tracks = []
    else:
        state = dict(last_bbox=None,
                     class_hist=deque(maxlen=10),
                     conf_hist=deque(maxlen=10))

    print(f"Starting pipeline in {mode.upper()} mode | Averaging: {'ON' if avg_classes else 'OFF'}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        buffer.append(gray)
        if len(buffer) <= params["k2"]:
            continue

        if mode == "multi":
            frame_out, fgMask, tracks = process_frame_multi(frame, gray, buffer, tracks, params, avg_classes)
        else:
            frame_out, fgMask, state = process_frame_mono(frame, gray, buffer, state, params, avg_classes)

        txt = mode.upper() + " - VIDEO: " + str(videos[sel])
        cv2.putText(frame_out, txt, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)    
        cv2.imshow("Frame + Detections", frame_out)
        cv2.imshow("Foreground Mask", fgMask)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def updateParams(mono, params):
    if mono:
        params["min_area"] = 70
        params["threshold"] = 5
    else:
        params["min_area"] = 150
        params["threshold"] = 20
    return params


# ============================================================
# Run pipeline
# ============================================================
if __name__ == "__main__":
    vid_dir = "assets/videos/"
    videos = ["V_DRONE_001", "V_DRONE_027", "V_DRONE_048",
              "V_BIRD_019", "V_BIRD_045", "V_BIRD_046"]
    sel = 1
    mono = False
    mode = "mono" if mono else "multi"
    avg_classes = True

    video_path = vid_dir + videos[sel] + ".mp4"

    params = dict(
        k1=1, k2=4, threshold=5, min_area=100,
        max_match_dist=40, timeout_max=5, overlap_thresh=0.01
    )

    params = updateParams(mono, params)
    run_video_pipeline(video_path, params, mode, avg_classes)
