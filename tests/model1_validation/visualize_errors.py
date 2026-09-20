from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path("ai/tests/model1_validation/visdrone_yolo")
IMG_DIR = ROOT / "images"
LBL_DIR = ROOT / "labels"

OUT = Path("ai/tests/model1_validation/outputs/error_visualizations")
FP_OUT = OUT / "false_positives"
FN_OUT = OUT / "false_negatives"

FP_OUT.mkdir(parents=True, exist_ok=True)
FN_OUT.mkdir(parents=True, exist_ok=True)

MODEL = "ai/models/person/best.pt"
CONF = 0.25
IOU_THRESHOLD = 0.5

model = YOLO(MODEL)


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)

    inter = iw * ih

    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    union = area_a + area_b - inter

    return inter / union if union else 0


for idx, image_path in enumerate(sorted(IMG_DIR.glob("*")), 1):

    img = cv2.imread(str(image_path))

    if img is None:
        continue

    h, w = img.shape[:2]

    label_path = LBL_DIR / f"{image_path.stem}.txt"

    gt = []

    if label_path.exists():
        for line in label_path.read_text().splitlines():

            if not line.strip():
                continue

            p = line.split()

            if len(p) < 5:
                continue

            _, xc, yc, bw, bh = map(float, p[:5])

            gt.append([
                (xc - bw / 2) * w,
                (yc - bh / 2) * h,
                (xc + bw / 2) * w,
                (yc + bh / 2) * h
            ])

    result = model.predict(
        source=str(image_path),
        conf=CONF,
        imgsz=640,
        verbose=False
    )[0]

    preds = []

    if result.boxes is not None:
        for box, conf in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy()
        ):
            preds.append([
                float(box[0]),
                float(box[1]),
                float(box[2]),
                float(box[3]),
                float(conf)
            ])

    matched_gt = set()
    matched_pred = set()

    for pi, pred in enumerate(preds):

        best_iou = 0
        best_gt = -1

        for gi, truth in enumerate(gt):

            if gi in matched_gt:
                continue

            score = iou(pred[:4], truth)

            if score > best_iou:
                best_iou = score
                best_gt = gi

        if best_iou >= IOU_THRESHOLD:
            matched_pred.add(pi)
            matched_gt.add(best_gt)

    fp = [i for i in range(len(preds)) if i not in matched_pred]
    fn = [i for i in range(len(gt)) if i not in matched_gt]

    # Draw TRUE POSITIVE predictions in green
    for pi in matched_pred:
        x1, y1, x2, y2, conf = preds[pi]

        cv2.rectangle(
            img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            f"TP {conf:.2f}",
            (int(x1), max(15, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    # Draw FALSE POSITIVE predictions in red
    for pi in fp:
        x1, y1, x2, y2, conf = preds[pi]

        cv2.rectangle(
            img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 255),
            2
        )

        cv2.putText(
            img,
            f"FP {conf:.2f}",
            (int(x1), max(15, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

    # Draw MISSED ground-truth people in yellow
    for gi in fn:
        x1, y1, x2, y2 = gt[gi]

        cv2.rectangle(
            img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 255),
            2
        )

        cv2.putText(
            img,
            "FN",
            (int(x1), max(15, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )

    # Add summary
    cv2.rectangle(img, (0, 0), (430, 35), (0, 0, 0), -1)

    text = f"TP: {len(matched_pred)}  FP: {len(fp)}  FN: {len(fn)}"

    cv2.putText(
        img,
        text,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # Save FP/FN images
    if fp:
        cv2.imwrite(
            str(FP_OUT / image_path.name),
            img
        )

    if fn:
        cv2.imwrite(
            str(FN_OUT / image_path.name),
            img
        )

    if idx % 50 == 0:
        print(f"Processed {idx}/{len(list(IMG_DIR.glob('*')))}")

print("\nDONE")
print("FP visualizations:", FP_OUT)
print("FN visualizations:", FN_OUT)
