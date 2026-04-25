import numpy as np

MIN_HISTORY_POINTS = 6


def path_consistency_cost(center_history, predicted_point, candidate_point):
    """
    KF-aware path consistency cost.

    Inputs:
      - center_history: recent BEV trajectory points of one track
      - predicted_point: KF-predicted BEV position at current frame
      - candidate_point: detection BEV position at current frame

    The candidate is compared against the KF prediction in the local path frame
    estimated from history. A large residual along path-normal direction leads
    to high cost.
    """
    if (
        center_history is None
        or predicted_point is None
        or candidate_point is None
        or len(center_history) < MIN_HISTORY_POINTS
    ):
        return 0.0

    history = np.asarray([(x, y) for _, x, y in center_history], dtype=float)
    predicted = np.asarray(predicted_point, dtype=float)
    candidate = np.asarray(candidate_point, dtype=float)
    if (
        history.ndim != 2
        or history.shape[1] != 2
        or not np.all(np.isfinite(predicted))
        or not np.all(np.isfinite(candidate))
    ):
        return 0.0

    increments = np.diff(history, axis=0)
    if len(increments) < 2:
        return 0.0

    eps = 1e-6

    # Estimate the local path tangent from recent points via PCA, then orient it
    # according to the net displacement of the track.
    centered = history - np.mean(history, axis=0, keepdims=True)
    point_cov = centered.T @ centered / max(len(history) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(point_cov)
    order = np.argsort(eigvals)
    lambda_minor = float(max(eigvals[order[0]], 0.0))
    lambda_major = float(max(eigvals[order[-1]], 0.0))
    tangent = eigvecs[:, order[-1]]

    net_disp = history[-1] - history[0]
    if np.dot(tangent, net_disp) < 0:
        tangent = -tangent
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < eps:
        return 0.0
    tangent = tangent / tangent_norm
    normal = np.asarray([-tangent[1], tangent[0]], dtype=float)

    path_length = float(np.sum(np.linalg.norm(increments, axis=1)))
    net_length = float(np.linalg.norm(net_disp))
    straightness = net_length / (path_length + eps)
    anisotropy = (lambda_major - lambda_minor) / (lambda_major + lambda_minor + eps)
    reliability = float(np.clip(straightness * anisotropy, 0.0, 1.0))
    if reliability <= 0.0:
        return 0.0

    hist_local = np.column_stack([increments @ tangent, increments @ normal])
    residual_xy = candidate - predicted
    residual_local = np.asarray([residual_xy @ tangent, residual_xy @ normal], dtype=float)

    # Use local increment covariance as adaptive scaling.
    cov = np.cov(hist_local, rowvar=False)
    if cov.shape == ():
        cov = np.eye(2, dtype=float) * float(cov)
    cov = np.asarray(cov, dtype=float)

    # Numerical + scale regularization:
    # when local motion is almost deterministic (near-zero covariance), use a
    # floor derived from recent step length to avoid pathological saturation.
    step_scale = float(np.median(np.linalg.norm(increments, axis=1)))
    reg = max(float(np.trace(cov)) * 1e-3, (0.25 * step_scale) ** 2, eps)
    cov = cov + np.eye(2, dtype=float) * reg

    residual = residual_local
    try:
        maha = float(residual.T @ np.linalg.inv(cov) @ residual)
    except np.linalg.LinAlgError:
        maha = float(residual.T @ np.linalg.pinv(cov) @ residual)

    cost = 1.0 - np.exp(-0.5 * max(maha, 0.0))
    return float(np.clip(reliability * cost, 0.0, 1.0))


def path_consistency_distance(tracks, detections):
    """Return pairwise path-consistency costs for tracks and detections."""
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=float)

    dists = np.zeros((len(tracks), len(detections)), dtype=float)
    for row, track in enumerate(tracks):
        history = getattr(track, "center_history", None)
        predicted = None
        if getattr(track, "mean", None) is not None:
            mean = np.asarray(track.mean, dtype=float)
            if mean.shape[0] >= 2 and np.all(np.isfinite(mean[:2])):
                predicted = mean[:2].copy()
        if predicted is None and history is not None and len(history) > 0:
            _, px, py = history[-1]
            predicted = np.asarray([px, py], dtype=float)
        for col, det in enumerate(detections):
            candidate = getattr(det, "motion_xy", None)
            dists[row, col] = path_consistency_cost(history, predicted, candidate)
    return dists
