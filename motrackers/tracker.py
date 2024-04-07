from collections import OrderedDict
import numpy as np
from scipy.spatial import distance
from motrackers.utils.misc import get_centroid
from motrackers.track import Track


class Tracker:

    def __init__(self, max_lost=5, tracker_output_format='mot_challenge'):
        self.next_track_id = 0
        self.tracks = OrderedDict()
        self.max_lost = max_lost
        self.frame_count = 0
        self.tracker_output_format = tracker_output_format

    def _add_track(self, frame_id, bbox, detection_confidence, class_id, **kwargs):

        self.tracks[self.next_track_id] = Track(
            self.next_track_id, frame_id, bbox, detection_confidence, class_id=class_id,
            data_output_format=self.tracker_output_format,
            **kwargs
        )
        self.next_track_id += 1

    def _remove_track(self, track_id):

        del self.tracks[track_id]

    def _update_track(self, track_id, frame_id, bbox, detection_confidence, class_id, lost=0, iou_score=0., **kwargs):
    
        self.tracks[track_id].update(
            frame_id, bbox, detection_confidence, class_id=class_id, lost=lost, iou_score=iou_score, **kwargs
        )

    @staticmethod
    def _get_tracks(tracks):
        
        outputs = []
        for _, track in tracks.items():
            # if not track.lost:
            #     outputs.append(track.output())
            outputs.append(track.output())
        return outputs

    @staticmethod
    def preprocess_input(bboxes, class_ids, detection_scores):

        new_bboxes = np.array(bboxes, dtype='float')
        new_class_ids = np.array(class_ids, dtype='int')
        new_detection_scores = np.array(detection_scores)

        new_detections = list(zip(new_bboxes, new_class_ids, new_detection_scores))
        return new_detections

    def update(self, bboxes, detection_scores, class_ids):

        self.frame_count += 1

        if len(bboxes) == 0:
            lost_ids = list(self.tracks.keys())

            for track_id in lost_ids:
                self.tracks[track_id].lost += 1
                if self.tracks[track_id].lost > self.max_lost:
                    self._remove_track(track_id)

            outputs = self._get_tracks(self.tracks)
            return outputs

        detections = Tracker.preprocess_input(bboxes, class_ids, detection_scores)

        track_ids = list(self.tracks.keys())

        updated_tracks, updated_detections = [], []

        if len(track_ids):
            track_centroids = np.array([self.tracks[tid].centroid for tid in track_ids])
            detection_centroids = get_centroid(np.asarray(bboxes))

            centroid_distances = distance.cdist(track_centroids, detection_centroids)

            track_indices = np.amin(centroid_distances, axis=1).argsort()

            for idx in track_indices:
                track_id = track_ids[idx]

                remaining_detections = [
                    (i, d) for (i, d) in enumerate(centroid_distances[idx, :]) if i not in updated_detections]

                if len(remaining_detections):
                    detection_idx, detection_distance = min(remaining_detections, key=lambda x: x[1])
                    bbox, class_id, confidence = detections[detection_idx]
                    self._update_track(track_id, self.frame_count, bbox, confidence, class_id=class_id)
                    updated_detections.append(detection_idx)
                    updated_tracks.append(track_id)

                if len(updated_tracks) == 0 or track_id is not updated_tracks[-1]:
                    self.tracks[track_id].lost += 1
                    if self.tracks[track_id].lost > self.max_lost:
                        self._remove_track(track_id)

        for i, (bbox, class_id, confidence) in enumerate(detections):
            if i not in updated_detections:
                self._add_track(self.frame_count, bbox, confidence, class_id=class_id)

        outputs = self._get_tracks(self.tracks)
        return outputs
