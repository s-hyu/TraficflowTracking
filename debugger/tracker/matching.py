import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from . import kalman_filter
from .path_consistency import path_consistency_distance
import time

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def center_distance(atracks, btracks):
    """
    Compute pairwise center L2 distance.
    :type atracks: list[STrack] | np.ndarray (tlbr)
    :type btracks: list[STrack] | np.ndarray (tlbr)
    :rtype cost_matrix np.ndarray
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
        a_centers = np.stack([(atlbrs[:, 0] + atlbrs[:, 2]) * 0.5, (atlbrs[:, 1] + atlbrs[:, 3]) * 0.5], axis=1)
        b_centers = np.stack([(btlbrs[:, 0] + btlbrs[:, 2]) * 0.5, (btlbrs[:, 1] + btlbrs[:, 3]) * 0.5], axis=1)
    else:
        if len(atracks) == 0 or len(btracks) == 0:
            return np.zeros((len(atracks), len(btracks)), dtype=float)
        a_tlwh = np.asarray([t.tlwh for t in atracks], dtype=float)
        b_tlwh = np.asarray([t.tlwh for t in btracks], dtype=float)
        a_centers = np.stack([a_tlwh[:, 0] + a_tlwh[:, 2] * 0.5, a_tlwh[:, 1] + a_tlwh[:, 3] * 0.5], axis=1)
        b_centers = np.stack([b_tlwh[:, 0] + b_tlwh[:, 2] * 0.5, b_tlwh[:, 1] + b_tlwh[:, 3] * 0.5], axis=1)
    return cdist(a_centers, b_centers, metric="euclidean")


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def mahalanobis_distance(kf, tracks, detections, only_position=True):
    """
    Compute pairwise squared Mahalanobis distance between KF-predicted tracks
    and current detections in measurement space.
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=float)

    measurements = np.asarray([det.to_xyah() for det in detections], dtype=float)
    dists = np.full((len(tracks), len(detections)), np.inf, dtype=float)
    for i, track in enumerate(tracks):
        if track.mean is None or track.covariance is None:
            continue
        dists[i] = kf.gating_distance(
            track.mean,
            track.covariance,
            measurements,
            only_position=only_position,
            metric='maha',
        )
    return dists


def normalized_position_distance(kf, tracks, detections, maha_thresh, maha_thresh_roi=None, maha_roi_polygon=None, only_position=True):
    """Return Mahalanobis position cost normalized to [0, 1]."""
    if kf is None or maha_thresh is None or maha_thresh <= 0:
        return np.zeros((len(tracks), len(detections)), dtype=float)
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=float)

    maha_dists = mahalanobis_distance(kf, tracks, detections, only_position=only_position)
    det_maha_thresh = _build_detection_maha_thresholds(
        detections,
        maha_thresh=maha_thresh,
        maha_thresh_roi=maha_thresh_roi,
        maha_roi_polygon=maha_roi_polygon,
    )
    det_maha_thresh = np.maximum(det_maha_thresh, 1e-6)
    return np.clip(maha_dists / det_maha_thresh[None, :], 0.0, 1.0)


def _build_debug_records(
    frame_id,
    stage,
    tracks,
    detections,
    matches,
    final_dists,
    reid_dists,
    position_dists,
    path_dists,
    debug_track_ids=None,
):
    records = []
    if len(detections) == 0:
        return records

    track_ids = [getattr(track, "track_id", -1) for track in tracks]
    match_map = {int(det_idx): int(track_idx) for track_idx, det_idx in matches.tolist()} if len(matches) > 0 else {}
    track_filter = set(debug_track_ids or [])

    for det_idx, det in enumerate(detections):
        selected_row = match_map.get(det_idx, None)
        best_row = None
        if final_dists.size > 0 and final_dists.shape[0] > 0:
            col = final_dists[:, det_idx]
            finite_rows = np.where(np.isfinite(col))[0]
            if len(finite_rows) > 0:
                best_row = int(finite_rows[np.argmin(col[finite_rows])])
            elif col.size > 0:
                best_row = int(np.argmin(col))
        if selected_row is None:
            selected_row = best_row
        if selected_row is None:
            continue

        selected_track_id = int(track_ids[selected_row]) if selected_row < len(track_ids) else -1
        best_track_id = int(track_ids[best_row]) if best_row is not None and best_row < len(track_ids) else -1
        if track_filter and selected_track_id not in track_filter and best_track_id not in track_filter:
            continue

        tlwh = det.tlwh.copy().tolist()
        motion_xy = getattr(det, "motion_xy", None)
        if motion_xy is not None and np.all(np.isfinite(motion_xy)):
            bev_x = float(motion_xy[0])
            bev_y = float(motion_xy[1])
        else:
            bev_x = float("nan")
            bev_y = float("nan")
        records.append(
            {
                "frame": int(frame_id) if frame_id is not None else -1,
                "stage": stage,
                "det_index": int(det_idx),
                "matched": det_idx in match_map,
                "selected_track_id": selected_track_id,
                "best_track_id": best_track_id,
                "appearance_cost": float(reid_dists[selected_row, det_idx]) if reid_dists is not None and reid_dists.size else float("nan"),
                "position_cost": float(position_dists[selected_row, det_idx]) if position_dists is not None and position_dists.size else float("nan"),
                "path_cost": float(path_dists[selected_row, det_idx]) if path_dists is not None and path_dists.size else float("nan"),
                "final_cost": float(final_dists[selected_row, det_idx]) if final_dists is not None and final_dists.size else float("nan"),
                "tlwh": tlwh,
                "bev_x": bev_x,
                "bev_y": bev_y,
            }
        )

    return records


def build_overlay_annotations(records, fields=None):
    fields = list(fields or ["status", "track_id", "final_cost"])

    def _format_value(row, field):
        if field == "status":
            return "M" if row["matched"] else "B"
        if field == "track_id":
            track_id = row["selected_track_id"] if row["selected_track_id"] >= 0 else row["best_track_id"]
            return f"T{track_id}"
        if field == "det_index":
            return f"d{row['det_index']}"
        if field == "stage":
            return row["stage"]
        if field == "appearance_cost":
            return f"a={row['appearance_cost']:.2f}"
        if field == "position_cost":
            return f"p={row['position_cost']:.2f}"
        if field == "path_cost":
            return f"path={row['path_cost']:.2f}"
        if field == "final_cost":
            return f"f={row['final_cost']:.2f}"
        return None

    annotations = []
    for row in records:
        pieces = []
        for field in fields:
            value = _format_value(row, field)
            if value is not None:
                pieces.append(value)
        if not pieces:
            continue

        lines = []
        current = []
        for piece in pieces:
            current.append(piece)
            if len(current) == 3:
                lines.append(" ".join(current))
                current = []
        if current:
            lines.append(" ".join(current))

        annotations.append(
            {
                "tlwh": row["tlwh"],
                "matched": row["matched"],
                "lines": lines,
            }
        )
    return annotations


def _parse_polygon(polygon):
    if polygon is None:
        return None

    if isinstance(polygon, str):
        polygon = polygon.strip()
        if polygon == "":
            return None
        points = []
        for part in polygon.split(";"):
            part = part.strip()
            if not part:
                continue
            xy = [v.strip() for v in part.split(",")]
            if len(xy) != 2:
                continue
            try:
                points.append((float(xy[0]), float(xy[1])))
            except ValueError:
                continue
        if len(points) < 3:
            return None
        return np.asarray(points, dtype=np.float32)

    arr = np.asarray(polygon, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size < 6 or arr.size % 2 != 0:
            return None
        arr = arr.reshape(-1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
        return None
    return arr


def _build_detection_maha_thresholds(detections, maha_thresh, maha_thresh_roi=None, maha_roi_polygon=None):
    det_count = len(detections)
    if det_count == 0:
        return np.zeros((0,), dtype=np.float32)

    base = float(maha_thresh)
    det_thresh = np.full((det_count,), base, dtype=np.float32)
    if maha_thresh_roi is None:
        return det_thresh

    roi_val = float(maha_thresh_roi)
    if roi_val <= 0:
        return det_thresh

    polygon = _parse_polygon(maha_roi_polygon)
    if polygon is None:
        return det_thresh

    for i, det in enumerate(detections):
        tlwh = det.tlwh
        cx = float(tlwh[0] + 0.5 * tlwh[2])
        cy = float(tlwh[1] + 0.5 * tlwh[3])
        if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
            det_thresh[i] = roi_val
    return det_thresh


def associate_tracks_detections(
    tracks,
    detections,
    match_thresh,
    kf=None,
    maha_thresh=None,
    maha_thresh_roi=None,
    maha_roi_polygon=None,
    use_maha_gate=True,
    only_position=True,
    mot20=False,
    use_fuse_score_on_iou=True,
    frame_id=None,
    velocity_min_speed=1.0,
    debug_records=None,
    debug_stage="",
    debug_track_ids=None,
):
    """
    Unified association entry:
    1) ReID (if features available) + BEV path consistency debug cost
    2) Otherwise IoU distance (+ optional score fusion)
    3) Hungarian assignment with threshold
    """
    use_reid = (
        len(tracks) > 0
        and len(detections) > 0
        and all(t.smooth_feat is not None for t in tracks)
        and all(d.curr_feat is not None for d in detections)
    )

    reid_dists = None
    path_dists = None
    position_dists = None

    if use_reid:
        reid_dists = np.clip(embedding_distance(tracks, detections), 0.0, 1.0)
        path_dists = path_consistency_distance(tracks, detections)
        position_dists = normalized_position_distance(
            kf,
            tracks,
            detections,
            maha_thresh=maha_thresh,
            maha_thresh_roi=maha_thresh_roi,
            maha_roi_polygon=maha_roi_polygon,
            only_position=only_position,
        )
        dists = 0.4 * reid_dists + 0.4 * path_dists + 0.2 * position_dists
        if use_maha_gate and kf is not None and maha_thresh is not None and maha_thresh > 0:
            maha_dists = mahalanobis_distance(kf, tracks, detections, only_position=only_position)
            det_maha_thresh = _build_detection_maha_thresholds(
                detections,
                maha_thresh=maha_thresh,
                maha_thresh_roi=maha_thresh_roi,
                maha_roi_polygon=maha_roi_polygon,
            )
            invalid = maha_dists > det_maha_thresh[None, :]
            dists[invalid] = np.inf
    else:
        dists = iou_distance(tracks, detections)
        if use_fuse_score_on_iou and not mot20:
            dists = fuse_score(dists, detections)

    matches, unmatched_tracks, unmatched_dets = linear_assignment(dists, thresh=match_thresh)
    if debug_records is not None:
        debug_records.extend(
            _build_debug_records(
                frame_id=frame_id,
                stage=debug_stage,
                tracks=tracks,
                detections=detections,
                matches=matches,
                final_dists=dists,
                reid_dists=reid_dists,
                position_dists=position_dists,
                path_dists=path_dists,
                debug_track_ids=debug_track_ids,
            )
        )

    return matches, unmatched_tracks, unmatched_dets
