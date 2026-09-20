from ultralytics import YOLO
from pathlib import Path
import cv2
import time

MODEL = "ai/models/person_visdrone/best.pt"
ROOT = Path("ai/tests/model1_validation/visdrone/VisDrone2019-DET-val")
IMAGES = ROOT / "images"
ANN = ROOT / "annotations"

model = YOLO(MODEL)

# VisDrone small-object definition:
# area < 32x32 pixels
SMALL_AREA = 32 * 32

TP = FP = FN = 0
GT_SMALL = 0
PRED_SMALL = 0

def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2-x1) * max(0, y2-y1)

    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])

    union = area_a + area_b - inter

    return inter / union if union else 0


for image_path in sorted(IMAGES.glob("*.jpg")):

    ann_path = ANN / f"{image_path.stem}.txt"

    gt = []

    with open(ann_path) as f:
        for line in f:
            p = line.strip().split(",")

            if len(p) < 6:
                continue

            x, y, w, h, score, cls = map(int, p[:6])

            # VisDrone:
            # 1 = pedestrian
            # 2 = people
            if cls not in (1, 2):
                continue

            area = w * h

            if area < SMALL_AREA:
                gt.append([x, y, x+w, y+h])
                GT_SMALL += 1

    result = model.predict(
        source=str(image_path),
        imgsz=1280,
        conf=0.20,
        iou=0.45,
        device=0,
        verbose=False
    )[0]

    preds = []

    if result.boxes is not None:

        for box, cls in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy()
        ):

            if int(cls) not in (0, 1):
                continue

            b = box.tolist()

            area = max(0, b[2]-b[0]) * max(0, b[3]-b[1])

            if area < SMALL_AREA:
                preds.append(b)
                PRED_SMALL += 1

    matched = set()

    for pred in preds:

        best = 0
        best_idx = -1

        for i, target in enumerate(gt):

            if i in matched:
                continue

            score = iou(pred, target)

            if score > best:
                best = score
                best_idx = i

        if best >= 0.5:
            TP += 1
            matched.add(best_idx)
        else:
            FP += 1

    FN += len(gt) - len(matched)


precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

print("\n" + "="*55)
print("SMALL PERSON EVALUATION — VISDRONE YOLOv8m")
print("="*55)
print(f"Small GT persons : {GT_SMALL}")
print(f"Small predictions : {PRED_SMALL}")
print(f"TP               : {TP}")
print(f"FP               : {FP}")
print(f"FN               : {FN}")
print()
print(f"Precision        : {precision:.4f}")
print(f"Recall           : {recall:.4f}")
print(f"F1               : {f1:.4f}")
print("="*55)
