from pathlib import Path
import shutil
import csv
from ultralytics import YOLO

ROOT = Path("ai/tests/model1_validation/visdrone_yolo")
IMG_DIR = ROOT / "images"
LBL_DIR = ROOT / "labels"

OUT = Path("ai/tests/model1_validation/outputs")
FP_DIR = OUT / "false_positives"
FN_DIR = OUT / "false_negatives"
TP_DIR = OUT / "true_positives"

for d in [FP_DIR, FN_DIR, TP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

model = YOLO("ai/models/person/best.pt")

IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.25

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0, box1[2]-box1[0]) * max(0, box1[3]-box1[1])
    area2 = max(0, box2[2]-box2[0]) * max(0, box2[3]-box2[1])

    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


rows = []

total_gt = 0
total_pred = 0
total_tp = 0
total_fp = 0
total_fn = 0

images = sorted(IMG_DIR.glob("*"))

for idx, image_path in enumerate(images, 1):

    label_path = LBL_DIR / f"{image_path.stem}.txt"

    # Read ground truth
    gt = []

    if label_path.exists():
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue

            p = line.split()

            if len(p) < 5:
                continue

            _, xc, yc, w, h = map(float, p[:5])

            gt.append([
                xc - w/2,
                yc - h/2,
                xc + w/2,
                yc + h/2
            ])

    # Run model
    result = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        imgsz=640,
        verbose=False
    )[0]

    preds = []

    if result.boxes is not None:
        for box, conf in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy()
        ):
            x1, y1, x2, y2 = box

            # Convert prediction to normalized coordinates
            iw, ih = result.orig_shape[1], result.orig_shape[0]

            preds.append([
                x1/iw,
                y1/ih,
                x2/iw,
                y2/ih,
                float(conf)
            ])

    matched_gt = set()
    matched_pred = set()

    for pi, pred in enumerate(preds):

        best_iou = 0
        best_gi = -1

        for gi, truth in enumerate(gt):

            if gi in matched_gt:
                continue

            score = iou(pred[:4], truth)

            if score > best_iou:
                best_iou = score
                best_gi = gi

        if best_iou >= IOU_THRESHOLD:
            matched_pred.add(pi)
            matched_gt.add(best_gi)

    tp = len(matched_pred)
    fp = len(preds) - tp
    fn = len(gt) - len(matched_gt)

    total_gt += len(gt)
    total_pred += len(preds)
    total_tp += tp
    total_fp += fp
    total_fn += fn

    # Copy FP images
    if fp > 0:
        shutil.copy2(image_path, FP_DIR / image_path.name)

    # Copy FN images
    if fn > 0:
        shutil.copy2(image_path, FN_DIR / image_path.name)

    # Copy TP images
    if tp > 0:
        shutil.copy2(image_path, TP_DIR / image_path.name)

    rows.append([
        image_path.name,
        len(gt),
        len(preds),
        tp,
        fp,
        fn
    ])

    if idx % 25 == 0:
        print(f"Processed {idx}/{len(images)}")

# Metrics
precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0
recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0

csv_path = OUT / "model1_error_analysis.csv"

with csv_path.open("w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "image",
        "ground_truth",
        "predictions",
        "true_positives",
        "false_positives",
        "false_negatives"
    ])

    writer.writerows(rows)

print("\n========== MODEL 1 ERROR ANALYSIS ==========")
print(f"Images             : {len(images)}")
print(f"Ground truth       : {total_gt}")
print(f"Predictions        : {total_pred}")
print(f"True positives     : {total_tp}")
print(f"False positives    : {total_fp}")
print(f"False negatives    : {total_fn}")
print(f"Precision          : {precision:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"FP images          : {len(list(FP_DIR.iterdir()))}")
print(f"FN images          : {len(list(FN_DIR.iterdir()))}")
print(f"\nCSV: {csv_path}")
