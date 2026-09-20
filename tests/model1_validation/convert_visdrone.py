from pathlib import Path
import shutil

ROOT = Path("ai/tests/model1_validation/visdrone/VisDrone2019-DET-val")
IMG_SRC = ROOT / "images"
ANN_SRC = ROOT / "annotations"

OUT = Path("ai/tests/model1_validation/visdrone_yolo")
IMG_OUT = OUT / "images"
LBL_OUT = OUT / "labels"

IMG_OUT.mkdir(parents=True, exist_ok=True)
LBL_OUT.mkdir(parents=True, exist_ok=True)

images = list(IMG_SRC.glob("*"))

converted = 0
person_boxes = 0

for img in images:
    if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue

    ann = ANN_SRC / f"{img.stem}.txt"

    shutil.copy2(img, IMG_OUT / img.name)

    labels = []

    if ann.exists():
        w, h = None, None

        from PIL import Image
        with Image.open(img) as im:
            w, h = im.size

        for line in ann.read_text().splitlines():
            parts = line.strip().split(",")

            if len(parts) < 8:
                continue

            x, y, bw, bh = map(float, parts[:4])
            category = int(parts[5])

            # VisDrone:
            # 1 = pedestrian
            # 2 = people
            if category not in (1, 2):
                continue

            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h

            labels.append(
                f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"
            )
            person_boxes += 1

    (LBL_OUT / f"{img.stem}.txt").write_text(
        "\n".join(labels) + ("\n" if labels else "")
    )

    converted += 1

print(f"Images converted : {converted}")
print(f"Person boxes      : {person_boxes}")
print(f"Output            : {OUT}")
