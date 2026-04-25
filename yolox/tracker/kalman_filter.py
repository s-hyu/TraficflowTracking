# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking object motion.

    The 8-dimensional state space

        x, y, vx, vy, ax, ay, r, h

    contains the motion point position (x, y), point velocity (vx, vy),
    point acceleration (ax, ay), aspect ratio r, and height h. In the current
    vehicle-tracking pipeline, (x, y) can be the homography-projected
    bottom-center point in the ground-view plane while r/h remain image-box
    shape observations for compatibility.

    Object motion follows a constant acceleration model for the center point.
    The bounding box measurement (x, y, r, h) is taken as a direct observation
    of the state space (linear observation model).

    For vehicle tracking we additionally smooth vx/vy and ax/ay from a short
    observation window so the predicted center motion is less sensitive to
    frame-to-frame box jitter.

    """

    def __init__(self):
        dt = 1.0
        self._motion_mat = np.eye(8, 8)
        self._motion_mat[0, 2] = dt
        self._motion_mat[1, 3] = dt
        self._motion_mat[0, 4] = 0.5 * dt * dt
        self._motion_mat[1, 5] = 0.5 * dt * dt
        self._motion_mat[2, 4] = dt
        self._motion_mat[3, 5] = dt

        self._update_mat = np.zeros((4, 8))
        self._update_mat[0, 0] = 1.0
        self._update_mat[1, 1] = 1.0
        self._update_mat[2, 6] = 1.0
        self._update_mat[3, 7] = 1.0

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        self.velocity_weights = np.asarray([0.5, 1.0, 2.0, 3.0, 4.0], dtype=float)
        self.velocity_window = len(self.velocity_weights)
        self.center_window = self.velocity_window + 1

    def estimate_window_motion(self, center_history):
        """Estimate vx/vy and ax/ay from weighted per-frame center differences."""
        if center_history is None or len(center_history) < 2:
            return None

        history = list(center_history)[-self.center_window:]
        velocities = []
        velocity_frame_ids = []
        for idx in range(1, len(history)):
            prev_frame, prev_x, prev_y = history[idx - 1]
            curr_frame, curr_x, curr_y = history[idx]
            dt = max(int(curr_frame - prev_frame), 1)
            velocities.append(((curr_x - prev_x) / dt, (curr_y - prev_y) / dt))
            velocity_frame_ids.append(curr_frame)

        if not velocities:
            return None

        velocity_weights = self.velocity_weights[-len(velocities):]
        velocity_weights = velocity_weights / velocity_weights.sum()
        velocity_array = np.asarray(velocities, dtype=float)
        vx, vy = np.sum(velocity_array * velocity_weights[:, None], axis=0)

        ax = 0.0
        ay = 0.0
        if len(velocities) >= 2:
            accelerations = []
            for idx in range(1, len(velocities)):
                prev_vx, prev_vy = velocities[idx - 1]
                curr_vx, curr_vy = velocities[idx]
                dt = max(int(velocity_frame_ids[idx] - velocity_frame_ids[idx - 1]), 1)
                accelerations.append(((curr_vx - prev_vx) / dt, (curr_vy - prev_vy) / dt))
            accel_weights = self.velocity_weights[-len(accelerations):]
            accel_weights = accel_weights / accel_weights.sum()
            accel_array = np.asarray(accelerations, dtype=float)
            ax, ay = np.sum(accel_array * accel_weights[:, None], axis=0)

        return vx, vy, ax, ay

    def apply_window_motion(self, mean, center_history):
        """Overwrite center velocity/acceleration with smoothed window estimates."""
        motion = self.estimate_window_motion(center_history)
        if motion is None:
            return mean

        mean = mean.copy()
        mean[2] = motion[0]
        mean[3] = motion[1]
        mean[4] = motion[2]
        mean[5] = motion[3]
        return mean

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean = np.array([
            measurement[0],
            measurement[1],
            0.0,
            0.0,
            0.0,
            0.0,
            measurement[2],
            measurement[3],
        ], dtype=float)

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            20 * self._std_weight_velocity * measurement[3],
            20 * self._std_weight_velocity * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state.

        """
        std_pos = [
            self._std_weight_position * mean[7],
            self._std_weight_position * mean[7],
            self._std_weight_velocity * mean[7],
            self._std_weight_velocity * mean[7],
            2 * self._std_weight_velocity * mean[7],
            2 * self._std_weight_velocity * mean[7],
            1e-2,
            self._std_weight_position * mean[7],
        ]
        motion_cov = np.diag(np.square(std_pos))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * mean[7],
            self._std_weight_position * mean[7],
            1e-1,
            self._std_weight_position * mean[7],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state.
        """
        std_terms = [
            self._std_weight_position * mean[:, 7],
            self._std_weight_position * mean[:, 7],
            self._std_weight_velocity * mean[:, 7],
            self._std_weight_velocity * mean[:, 7],
            2 * self._std_weight_velocity * mean[:, 7],
            2 * self._std_weight_velocity * mean[:, 7],
            1e-2 * np.ones_like(mean[:, 7]),
            self._std_weight_position * mean[:, 7],
        ]
        sqr = np.square(np.asarray(std_terms, dtype=float)).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')
