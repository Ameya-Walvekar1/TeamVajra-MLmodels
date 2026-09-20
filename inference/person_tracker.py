from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Track:
    track_id: int
    bbox: List[float]
    confidence: float
    hits: int = 1
    missed: int = 0
    confirmed: bool = False


class PersonTracker:

    def __init__(
        self,
        iou_threshold=0.30,
        max_missed=5,
        confirm_hits=3,
    ):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.confirm_hits = confirm_hits

        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    @staticmethod
    def _iou(a, b):

        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        intersection = inter_w * inter_h

        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def update(self, detections):

        # detections:
        # [
        #   {
        #       "class": "PERSON",
        #       "source_class": "...",
        #       "confidence": float,
        #       "bbox": [x1,y1,x2,y2]
        #   }
        # ]

        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(self.tracks.keys())

        matches = []

        # Greedy highest-IoU matching
        candidates = []

        for track_id, track in self.tracks.items():

            for det_idx, detection in enumerate(detections):

                score = self._iou(
                    track.bbox,
                    detection["bbox"]
                )

                if score >= self.iou_threshold:
                    candidates.append(
                        (score, track_id, det_idx)
                    )

        candidates.sort(reverse=True)

        used_tracks = set()
        used_detections = set()

        for score, track_id, det_idx in candidates:

            if track_id in used_tracks:
                continue

            if det_idx in used_detections:
                continue

            used_tracks.add(track_id)
            used_detections.add(det_idx)

            matches.append(
                (track_id, det_idx)
            )

            unmatched_tracks.discard(track_id)
            unmatched_detections.discard(det_idx)

        # Update matched tracks
        for track_id, det_idx in matches:

            detection = detections[det_idx]
            track = self.tracks[track_id]

            track.bbox = detection["bbox"]
            track.confidence = detection["confidence"]
            track.hits += 1
            track.missed = 0

            if track.hits >= self.confirm_hits:
                track.confirmed = True

        # Age unmatched tracks
        for track_id in list(unmatched_tracks):

            track = self.tracks[track_id]
            track.missed += 1

            if track.missed > self.max_missed:
                del self.tracks[track_id]

        # Create new tracks
        for det_idx in unmatched_detections:

            detection = detections[det_idx]

            track = Track(
                track_id=self.next_id,
                bbox=detection["bbox"],
                confidence=detection["confidence"],
            )

            self.tracks[self.next_id] = track
            self.next_id += 1

        # Return active tracks
        output = []

        for track in self.tracks.values():

            # Only expose currently observed tracks
            # or tracks within the missed tolerance.
            if track.missed > self.max_missed:
                continue

            output.append({
                "track_id": track.track_id,
                "class": "PERSON",
                "bbox": track.bbox,
                "confidence": track.confidence,
                "hits": track.hits,
                "missed": track.missed,
                "confirmed": track.confirmed,
            })

        return output
