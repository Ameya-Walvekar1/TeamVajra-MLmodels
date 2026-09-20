from pathlib import Path
import csv
import cv2
from ultralytics import YOLO

ROOT = Path("ai/tests/model1_stress_test/VisDrone2019-VID-val")
MODEL_PATH = Path("models/person/best.pt")
OUT = Path("ai/tests/model1_stress_test/results")
OUT.mkdir(parents=True, exist_ok=True)

CONF = 0.30
NMS_IOU = 0.40
MATCH_IOU = 0.50

model = YOLO(str(MODEL_PATH))


def load_annotations(path):
    annotations = {}

    with open(path) as f:
        for line in f:
            p = line.strip().split(",")

            if len(p) < 10:
                continue

            frame = int(p[0])
            x = float(p[2])
            y = float(p[3])
            w = float(p[4])
            h = float(p[5])
            score = int(p[6])
            category = int(p[7])

            if category not in (1, 2):
                continue

            if score == 0:
                continue

            annotations.setdefault(frame, []).append(
                [x, y, x + w, y + h]
            )

    return annotations


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter

    return inter / union if union else 0


def size_category(box, image_w, image_h):
    x1, y1, x2, y2 = box

    area_ratio = (
        max(0, x2 - x1) *
        max(0, y2 - y1)
    ) / (image_w * image_h)

    # VisDrone-style practical relative-size buckets
    if area_ratio < 0.001:
        return "small"

    if area_ratio < 0.01:
        return "medium"

    return "large"


def evaluate(gt, pred):
    used = set()
    matched = []

    for gi, g in enumerate(gt):

        best = -1
        best_iou = 0

        for pi, p in enumerate(pred):

            if pi in used:
                continue

            score = iou(g, p)

            if score > best_iou:
                best_iou = score
                best = pi

        if best >= 0 and best_iou >= MATCH_IOU:
            used.add(best)
            matched.append((gi, best))

    return matched


annotations = {}

for f in (ROOT / "annotations").glob("*.txt"):
    annotations[f.stem] = load_annotations(f)

stats = {
    "small": {"gt": 0, "tp": 0},
    "medium": {"gt": 0, "tp": 0},
    "large": {"gt": 0, "tp": 0},
}

frames = sorted(ROOT.glob("sequences/*/*.jpg"))

print(f"Frames: {len(frames)}")
print(f"Confidence: {CONF}")
print(f"NMS IoU: {NMS_IOU}")
print()

for idx, image_path in enumerate(frames, 1):

    img = cv2.imread(str(image_path))

    if img is None:
        continue

    h, w = img.shape[:2]

    sequence = image_path.parent.name
    frame_id = int(image_path.stem)

    gt = annotations.get(sequence, {}).get(frame_id, [])

    result = model.predict(
        source=img,
        imgsz=640,
        conf=CONF,
        iou=NMS_IOU,
        verbose=False
    )[0]

    pred = [
        box.tolist()
        for box in result.boxes.xyxy.cpu()
    ]

    matches = evaluate(gt, pred)

    matched_gt = {x[0] for x in matches}

    for i, box in enumerate(gt):

        category = size_category(box, w, h)

        stats[category]["gt"] += 1

        if i in matched_gt:
            stats[category]["tp"] += 1

    if idx % 200 == 0:
        print(f"{idx}/{len(frames)}")

print()
print("=" * 60)
print("M1 SIZE-WISE RECALL")
print("=" * 60)

rows = []

for category in ["small", "medium", "large"]:

    gt = stats[category]["gt"]
    tp = stats[category]["tp"]

    recall = tp / gt if gt else 0

    rows.append({
        "size": category,
        "ground_truth": gt,
        "true_positive": tp,
        "false_negative": gt - tp,
        "recall": recall
    })

    print(
        f"{category.upper():8s} "
        f"GT={gt:6d} "
        f"TP={tp:6d} "
        f"FN={gt-tp:6d} "
        f"Recall={recall:.4f}"
    )

csv_path = OUT / "size_analysis.csv"

with open(csv_path, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(rows)

print()
print(f"Saved: {csv_path}")
