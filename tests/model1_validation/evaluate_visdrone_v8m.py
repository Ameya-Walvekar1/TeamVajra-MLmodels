from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import time

MODEL = "ai/models/person_visdrone/best.pt"
ROOT = Path("ai/tests/model1_validation/visdrone/VisDrone2019-DET-val")
IMAGES = ROOT / "images"
ANN = ROOT / "annotations"

model = YOLO(MODEL)

image_files = sorted(IMAGES.glob("*.jpg"))

print("=" * 60)
print("VISDRONE YOLOv8m — PERSON EVALUATION")
print("=" * 60)
print("Images:", len(image_files))
print("Model:", MODEL)
print("Classes:", model.names)
print()

# IoU calculation
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2-x1) * max(0, y2-y1)

    a1 = max(0, box1[2]-box1[0]) * max(0, box1[3]-box1[1])
    a2 = max(0, box2[2]-box2[0]) * max(0, box2[3]-box2[1])

    union = a1 + a2 - inter

    return inter / union if union > 0 else 0


# Store detections and GT
all_predictions = []
all_gt = []

total_time = 0
total_gt = 0
total_pred = 0

for idx, image_path in enumerate(image_files, 1):

    annotation_path = ANN / (image_path.stem + ".txt")

    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]

    # -----------------------------
    # Ground truth
    # -----------------------------
    gt = []

    if annotation_path.exists():
        with open(annotation_path) as f:
            for line in f:

                parts = line.strip().split(",")

                if len(parts) < 6:
                    continue

                x, y, bw, bh, score, cls = map(int, parts[:6])

                # Ignore regions
                if cls == 0:
                    continue

                # VisDrone person classes:
                # 1 = pedestrian
                # 2 = people
                if cls not in (1, 2):
                    continue

                gt_box = [
                    x,
                    y,
                    x + bw,
                    y + bh
                ]

                gt.append(gt_box)

    # -----------------------------
    # Prediction
    # -----------------------------
    t0 = time.perf_counter()

    result = model.predict(
        source=str(image_path),
        imgsz=1280,
        conf=0.20,
        iou=0.45,
        device=0,
        verbose=False
    )[0]

    total_time += time.perf_counter() - t0

    preds = []

    if result.boxes is not None:

        for box, conf, cls in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy()
        ):

            cls = int(cls)

            # Only pedestrian + people
            if cls not in (0, 1):
                continue

            preds.append((
                box.tolist(),
                float(conf)
            ))

    total_gt += len(gt)
    total_pred += len(preds)

    all_predictions.append(preds)
    all_gt.append(gt)

    if idx % 50 == 0:
        print(
            f"{idx:3d}/{len(image_files)} | "
            f"GT={total_gt} | "
            f"Pred={total_pred}"
        )


# ---------------------------------------------------------
# Calculate precision / recall at IoU 0.5
# ---------------------------------------------------------

TP = 0
FP = 0
FN = 0

for preds, gt in zip(all_predictions, all_gt):

    matched = set()

    # Highest confidence first
    preds = sorted(preds, key=lambda x: x[1], reverse=True)

    for pred_box, conf in preds:

        best_iou = 0
        best_idx = -1

        for j, gt_box in enumerate(gt):

            if j in matched:
                continue

            score = iou(pred_box, gt_box)

            if score > best_iou:
                best_iou = score
                best_idx = j

        if best_iou >= 0.5:
            TP += 1
            matched.add(best_idx)
        else:
            FP += 1

    FN += len(gt) - len(matched)


precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

latency = total_time / len(image_files) * 1000
fps = len(image_files) / total_time

print()
print("=" * 60)
print("RESULT")
print("=" * 60)

print(f"Ground-truth persons : {total_gt}")
print(f"Predicted persons    : {total_pred}")
print(f"True positives       : {TP}")
print(f"False positives      : {FP}")
print(f"False negatives      : {FN}")
print()
print(f"Precision             : {precision:.4f}")
print(f"Recall                : {recall:.4f}")
print(f"F1                    : {f1:.4f}")
print()
print(f"Average latency       : {latency:.2f} ms")
print(f"FPS                   : {fps:.2f}")
print("=" * 60)
