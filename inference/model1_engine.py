from ultralytics import YOLO
from inference.person_tracker import PersonTracker


class Model1PersonEngine:

    PERSON_CLASSES = {"pedestrian", "people"}

    def __init__(
        self,
        model_path,
        device=0,
        tracker_enabled=True,
        confirm_hits=3,
        max_missed=5,
        tracker_iou=0.30,
    ):

        self.model_path = str(model_path)
        self.device = device
        self.tracker_enabled = tracker_enabled

        self.model = YOLO(self.model_path)

        # Detect the person classes from the actual checkpoint.
        self.person_class_ids = [
            class_id
            for class_id, name in self.model.names.items()
            if name in self.PERSON_CLASSES
        ]

        if not self.person_class_ids:
            raise RuntimeError(
                "Model 1 does not contain pedestrian/people classes."
            )

        self.tracker = None

        if tracker_enabled:
            self.tracker = PersonTracker(
                iou_threshold=tracker_iou,
                max_missed=max_missed,
                confirm_hits=confirm_hits,
            )

    def predict(
        self,
        image,
        conf=0.20,
        iou=0.45,
        imgsz=1280,
    ):

        result = self.model.predict(
            source=image,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            classes=self.person_class_ids,
            device=self.device,
            verbose=False,
        )[0]

        detections = []

        if result.boxes is not None:

            boxes = result.boxes

            for box, confidence, class_id in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.conf.cpu().numpy(),
                boxes.cls.cpu().numpy(),
            ):

                class_id = int(class_id)

                detections.append({
                    "class": "PERSON",
                    "source_class": self.model.names[class_id],
                    "confidence": float(confidence),
                    "bbox": [float(v) for v in box],
                })

        # Tracking only makes sense for sequential frames.
        if self.tracker_enabled:
            tracks = self.tracker.update(detections)

            return tracks

        return detections

    def reset_tracker(self):

        if self.tracker is not None:
            self.tracker = PersonTracker(
                iou_threshold=self.tracker.iou_threshold,
                max_missed=self.tracker.max_missed,
                confirm_hits=self.tracker.confirm_hits,
            )
