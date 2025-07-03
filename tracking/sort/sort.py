import numpy as np
from scipy.optimize import linear_sum_assignment
from .kalman_filter import KalmanBoxTracker

def iou(bb_test, bb_gt):
    """
    Calcola IoU tra due bbox [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w*h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
              + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return o

class Sort:
    # modificare min hits max age e iou threshold per mitigare problema tracking multi-oggetto
    # max_age -> n max frame che un track può non essere associato a nessuna detection prima di essere eliminato
    # se un oggetto scompare, rimane in memoria per max_age frame
    # min_hits -> n min associamenti consecutivi che una traccia deve avere per essere considerata valida
    # filtrare falsi positivi
    #       valori originali max_age=5, min_hits=3, iou_threshold=0.3 

    def __init__(self, max_age=10, min_hits=1, iou_threshold=0.2):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, dets):
        """
        dets: [[x1,y1,x2,y2], ...]
        """
        # Predict
        trks = np.array([trk.predict() for trk in self.trackers])
        if len(trks) == 0:
            trks = np.empty((0, 4))
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # Update
        for t, trk in enumerate(self.trackers):
            if t in unmatched_trks:
                continue
            d = matched[matched[:,1] == t,0]
            if len(d) > 0:
                trk.update(dets[d[0]])

        # nuovi trackers
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i]))

        # Elimina tracker vecchi
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        # Ritorna bbox con id
        ret = []
        for trk in self.trackers:
            if trk.hits >= self.min_hits or trk.time_since_update == 0:
                ret.append(np.concatenate((trk.get_state(), [trk.id])))
        return np.array(ret)

def associate_detections_to_trackers(dets, trks, iou_threshold=0.3):
    if len(trks) == 0:
        return np.empty((0,2),dtype=int), np.arange(len(dets)), np.empty((0),dtype=int)

    iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
    for d, det in enumerate(dets):
        for t, trk in enumerate(trks):
            iou_matrix[d, t] = iou(det, trk)

    matched_indices = linear_sum_assignment(-iou_matrix)
    matched_indices = np.array(list(zip(*matched_indices)))

    unmatched_dets = []
    for d in range(len(dets)):
        if d not in matched_indices[:,0]:
            unmatched_dets.append(d)
    unmatched_trks = []
    for t in range(len(trks)):
        if t not in matched_indices[:,1]:
            unmatched_trks.append(t)

    # Remove low IoU matches
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if len(matches) == 0:
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return matches, np.array(unmatched_dets), np.array(unmatched_trks)
