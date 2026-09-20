from pathlib import Path
import cv2
import csv
import random
from ultralytics import YOLO

ROOT = Path("ai/tests/model1_stress_test/VisDrone2019-VID-val")
MODEL_PATH = Path("models/person/best.pt")
OUT = Path("ai/tests/model1_stress_test/results/error_visuals")

OUT.mkdir(parents=True, exist_ok=True)

CONF = 0.30
NMS_IOU = 0.40
MATCH_IOU = 0.50

MAX_IMAGES = 100

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


annotations = {}

for f in (ROOT / "annotations").glob("*.txt"):
    annotations[f.stem] = load_annotations(f)


def evaluate(gt, pred):
    used = set()
    matches = []

    for gi, g in enumerate(gt):

        best = -1
        best_score = 0

        for pi, p in enumerate(pred):

            if pi in used:
                continue

            score = iou(g, p)

            if score > best_score:
                best_score = score
                best = pi

        if best >= 0 and best_score >= MATCH_IOU:
            used.add(best)
            matches.append((gi, best))

    return matches


candidates = []

frames = sorted(ROOT.glob("sequences/*/*.jpg"))

print(f"Analysing {len(frames)} frames...")

for idx, image_path in enumerate(frames):

    img = cv2.imread(str(image_path))

    if img is None:
        continue

    h, w = img.shape[:2]

    sequence = image_path.parent.name
    frame_id = int(image_path.stem)

    gt_raw = annotations.get(sequence, {}).get(frame_id, [])

    gt = [
        [x1, y1, x2, y2]
        for x1, y1, x2, y2 in gt_raw
    ]

    result = model.predict(
        source=img,
        imgsz=640,
        conf=CONF,
        iou=NMS_IOU,
        verbose=False
    )[0]

    pred = []

    for box in result.boxes.xyxy.cpu().numpy():
        pred.append(box.tolist())

    matches = evaluate(gt, pred)

    matched_gt = {x[0] for x in matches}
    matched_pred = {x[1] for x in matches}

    fp = len(pred) - len(matched_pred)
    fn = len(gt) - len(matched_gt)

    # Prioritize frames with actual errors
    error_score = fp + fn

    if error_score > 0:

        candidates.append({
            "path": image_path,
            "fp": fp,
            "fn": fn,
            "score": error_score,
            "gt": gt,
            "pred": pred,
            "matched_gt": matched_gt,
            "matched_pred": matched_pred
        })


print(f"Candidate error frames: {len(candidates)}")

# Sort by most problematic frames
candidates.sort(
    key=lambda x: x["score"],
    reverse=True
)

selected = candidates[:MAX_IMAGES]

print(f"Saving top {len(selected)} error frames...")


for n, item in enumerate(selected, 1):

    image = cv2.imread(str(item["path"]))

    # Ground truth
    for i, box in enumerate(item["gt"]):

        x1, y1, x2, y2 = map(int, box)

        if i in item["matched_gt"]:
            thickness = 2
            color = (0, 255, 0)       # green
        else:
            thickness = 3
            color = (0, 255, 255)     # yellow

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

    # Predictions
    for i, box in enumerate(item["pred"]):

        x1, y1, x2, y2 = map(int, box)

        if i in item["matched_pred"]:
            continue

        color = (0, 0, 255)            # red

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

    label = (
        f"FP: {item['fp']} | "
        f"FN: {item['fn']} | "
        f"GT: {item['gt']} | "
        f"Pred: {len(item['pred'])}"
    )

    cv2.rectangle(
        image,
        (0, 0),
        (image.shape[1], 35),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        image,
        label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    filename = (
        f"{n:03d}_"
        f"fp{item['fp']}_"
        f"fn{item['fn']}_"
        f"{item['path'].parent.name}_"
        f"{item['path'].name}"
    )

    cv2.imwrite(
        str(OUT / filename),
        image
    )


print()
print("=" * 60)
print("ERROR VISUALIZATION COMPLETE")
print("=" * 60)
print(f"Output: {OUT}")
print()
print("GREEN  = correct detection")
print("RED    = false positive")
print("YELLOW = missed person")
