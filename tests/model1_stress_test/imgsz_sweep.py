from pathlib import Path
import csv
import time
import cv2
from ultralytics import YOLO

ROOT = Path("ai/tests/model1_stress_test/VisDrone2019-VID-val")
MODEL_PATH = Path("models/person/best.pt")
OUT = Path("ai/tests/model1_stress_test/results/imgsz_sweep")
OUT.mkdir(parents=True, exist_ok=True)

IMGSZ_VALUES = [640, 768, 960, 1280]

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


def match(gt, pred):
    used = set()
    tp = 0

    for g in gt:

        best = -1
        best_score = 0

        for i, p in enumerate(pred):

            if i in used:
                continue

            score = iou(g, p)

            if score > best_score:
                best_score = score
                best = i

        if best >= 0 and best_score >= MATCH_IOU:
            used.add(best)
            tp += 1

    fp = len(pred) - tp
    fn = len(gt) - tp

    return tp, fp, fn


annotations = {}

for f in (ROOT / "annotations").glob("*.txt"):
    annotations[f.stem] = load_annotations(f)

frames = sorted(ROOT.glob("sequences/*/*.jpg"))

print(f"Frames: {len(frames)}")
print()

results = []

for imgsz in IMGSZ_VALUES:

    print("=" * 60)
    print(f"Testing imgsz={imgsz}")
    print("=" * 60)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_pred = 0

    start_total = time.perf_counter()

    for idx, image_path in enumerate(frames, 1):

        img = cv2.imread(str(image_path))

        if img is None:
            continue

        h, w = img.shape[:2]

        sequence = image_path.parent.name
        frame_id = int(image_path.stem)

        gt_raw = annotations.get(sequence, {}).get(frame_id, [])

        gt = [
            [x1 / w, y1 / h, x2 / w, y2 / h]
            for x1, y1, x2, y2 in gt_raw
        ]

        result = model.predict(
            source=img,
            imgsz=imgsz,
            conf=CONF,
            iou=NMS_IOU,
            verbose=False
        )[0]

        pred = []

        for box in result.boxes.xyxy.cpu().numpy():

            x1, y1, x2, y2 = box

            pred.append([
                x1 / w,
                y1 / h,
                x2 / w,
                y2 / h
            ])

        tp, fp, fn = match(gt, pred)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gt += len(gt)
        total_pred += len(pred)

        if idx % 400 == 0:
            print(f"{idx}/{len(frames)}")

    elapsed = time.perf_counter() - start_total

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp else 0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    fps = len(frames) / elapsed

    row = {
        "imgsz": imgsz,
        "gt": total_gt,
        "pred": total_pred,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fps": fps
    }

    results.append(row)

    print()
    print(
        f"imgsz={imgsz} | "
        f"P={precision:.4f} | "
        f"R={recall:.4f} | "
        f"F1={f1:.4f} | "
        f"FPS={fps:.2f}"
    )
    print()


csv_path = OUT / "imgsz_sweep_results.csv"

with open(csv_path, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)

print("=" * 60)
print("IMAGE SIZE SWEEP COMPLETE")
print("=" * 60)
print(f"Results: {csv_path}")
