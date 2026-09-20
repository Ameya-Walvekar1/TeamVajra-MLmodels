from pathlib import Path
import csv
import time
import cv2
from ultralytics import YOLO

ROOT = Path("ai/tests/model1_stress_test/VisDrone2019-VID-val")
MODEL_PATH = Path("models/person/best.pt")
OUT = Path("ai/tests/model1_stress_test/results")
OUT.mkdir(parents=True, exist_ok=True)

CONF = 0.25
IOU_MATCH = 0.50

model = YOLO(str(MODEL_PATH))


def load_sequence_annotations(path):
    """
    VisDrone VID:
    frame_id,target_id,x,y,width,height,score,category,truncation,occlusion
    """

    annotations = {}

    with open(path) as f:
        for line in f:
            p = line.strip().split(",")

            if len(p) < 10:
                continue

            frame_id = int(p[0])

            x = float(p[2])
            y = float(p[3])
            w = float(p[4])
            h = float(p[5])

            score = int(p[6])
            category = int(p[7])

            # 1 = pedestrian, 2 = people
            if category not in (1, 2):
                continue

            # score 0 = ignored
            if score == 0:
                continue

            annotations.setdefault(frame_id, []).append(
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

    return inter / union if union else 0.0


def match(gt, pred):
    used = set()
    tp = 0

    for g in gt:
        best_idx = -1
        best_iou = 0

        for i, p in enumerate(pred):
            if i in used:
                continue

            score = iou(g, p)

            if score > best_iou:
                best_iou = score
                best_idx = i

        if best_idx >= 0 and best_iou >= IOU_MATCH:
            used.add(best_idx)
            tp += 1

    fp = len(pred) - tp
    fn = len(gt) - tp

    return tp, fp, fn


def count_duplicates(pred):
    duplicates = 0

    for i in range(len(pred)):
        for j in range(i + 1, len(pred)):
            if iou(pred[i], pred[j]) >= 0.50:
                duplicates += 1

    return duplicates


# ---------------------------------------------------------
# Load all sequence annotations
# ---------------------------------------------------------

sequence_annotations = {}

for ann_file in (ROOT / "annotations").glob("*.txt"):
    sequence = ann_file.stem
    sequence_annotations[sequence] = load_sequence_annotations(ann_file)

print("Loaded annotation sequences:")
for seq, data in sequence_annotations.items():
    total = sum(len(v) for v in data.values())
    print(f"  {seq}: {len(data)} frames, {total} people")

# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------

rows = []

total_tp = 0
total_fp = 0
total_fn = 0
total_gt = 0
total_pred = 0
total_time = 0

frames = sorted(ROOT.glob("sequences/*/*.jpg"))

print()
print(f"Total frames: {len(frames)}")
print("Running M1 stress test...")

for idx, image_path in enumerate(frames, 1):

    img = cv2.imread(str(image_path))

    if img is None:
        continue

    h, w = img.shape[:2]

    sequence = image_path.parent.name

    # Frame filenames are zero-padded:
    # 0000001.jpg -> frame ID 1
    frame_id = int(image_path.stem)

    gt_original = sequence_annotations.get(sequence, {}).get(frame_id, [])

    # Convert GT to normalized coordinates
    gt = []

    for x1, y1, x2, y2 in gt_original:
        gt.append([
            x1 / w,
            y1 / h,
            x2 / w,
            y2 / h
        ])

    start = time.perf_counter()

    result = model.predict(
        source=img,
        imgsz=640,
        conf=CONF,
        verbose=False
    )[0]

    elapsed = time.perf_counter() - start
    total_time += elapsed

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

    duplicates = count_duplicates(pred)

    total_tp += tp
    total_fp += fp
    total_fn += fn
    total_gt += len(gt)
    total_pred += len(pred)

    rows.append({
        "sequence": sequence,
        "frame": image_path.name,
        "gt": len(gt),
        "pred": len(pred),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "duplicates": duplicates,
        "latency_ms": elapsed * 1000,
        "fps": 1 / elapsed if elapsed else 0
    })

    if idx % 200 == 0:
        print(f"{idx}/{len(frames)}")

# ---------------------------------------------------------
# Save CSV
# ---------------------------------------------------------

csv_path = OUT / "frame_results.csv"

with open(csv_path, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(rows)

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

precision = (
    total_tp / (total_tp + total_fp)
    if total_tp + total_fp else 0
)

recall = (
    total_tp / (total_tp + total_fn)
    if total_tp + total_fn else 0
)

avg_latency = (
    total_time / len(rows) * 1000
    if rows else 0
)

avg_fps = (
    len(rows) / total_time
    if total_time else 0
)

print()
print("=" * 55)
print("M1 STRESS TEST RESULT")
print("=" * 55)

print(f"Frames              : {len(rows)}")
print(f"Ground truth people : {total_gt}")
print(f"Predictions         : {total_pred}")
print(f"True positives      : {total_tp}")
print(f"False positives     : {total_fp}")
print(f"False negatives     : {total_fn}")
print(f"Precision           : {precision:.4f}")
print(f"Recall              : {recall:.4f}")
print(f"Average latency     : {avg_latency:.2f} ms")
print(f"Average FPS         : {avg_fps:.2f}")
print(f"Results CSV         : {csv_path}")

print("=" * 55)
