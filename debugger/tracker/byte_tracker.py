import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from . import matching
from .basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, feat=None, motion_xy=None):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.motion_xy = None if motion_xy is None else np.asarray(motion_xy, dtype=float)
        self.use_external_motion = self.motion_xy is not None
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.center_history = deque(maxlen=self.shared_kalman.center_window)

        self.curr_feat = None
        self.smooth_feat =None
        self.alpha = 0.9
        if feat is not None:
            self.update_features(feat)

    def _record_center(self, frame_id, tlwh=None):
        if frame_id is None:
            return
        if self.use_external_motion and self.motion_xy is not None and tlwh is None:
            x, y = self.motion_xy
        else:
            if tlwh is None:
                tlwh = self.tlwh
            xyah = self.tlwh_to_xyah(tlwh)
            x, y = xyah[:2]
        self.center_history.append((int(frame_id), float(x), float(y)))

    def _apply_window_velocity(self):
        if self.mean is None:
            return
        self.mean = self.shared_kalman.apply_window_motion(self.mean, self.center_history)

    def update_features(self,feat):
        feat = feat /(np.linalg.norm(feat) + 1e-8)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha  * self.smooth_feat + (1-self.alpha) * self.curr_feat
            self.smooth_feat = self.smooth_feat / (np.linalg.norm(self.smooth_feat)+ 1e-8)
    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[4] = 0
            mean_state[5] = 0
        mean_state = self.kalman_filter.apply_window_motion(mean_state, self.center_history)
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            for st in stracks:
                st._apply_window_velocity()
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][4] = 0
                    multi_mean[i][5] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.to_xyah())

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self._record_center(frame_id)

    def re_activate(self, new_track, frame_id, new_id=False):
        self._tlwh = new_track.tlwh.copy()
        self.motion_xy = None if new_track.motion_xy is None else new_track.motion_xy.copy()
        self.use_external_motion = new_track.use_external_motion
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.to_xyah()
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self._record_center(frame_id)
        self._apply_window_velocity()
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self._tlwh = new_tlwh.copy()
        self.motion_xy = None if new_track.motion_xy is None else new_track.motion_xy.copy()
        self.use_external_motion = new_track.use_external_motion
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.to_xyah())
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self._record_center(frame_id)
        self._apply_window_velocity()

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
            
    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.use_external_motion:
            return self._tlwh.copy()
        if self.mean is None:
            return self._tlwh.copy()
        x = self.mean[0]
        y = self.mean[1]
        r = self.mean[6]
        h = self.mean[7]
        w = r * h
        return np.asarray([x - w / 2, y - h / 2, w, h], dtype=float)

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        measurement = self.tlwh_to_xyah(self._tlwh)
        if self.use_external_motion and self.motion_xy is not None:
            measurement[0] = self.motion_xy[0]
            measurement[1] = self.motion_xy[1]
        return measurement

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        self.det_thresh = args.track_thresh
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.ground_homography = self._load_ground_homography(args)
        self.debug_costs = bool(getattr(args, "debug_costs", False))
        self.debug_frame_start = max(int(getattr(args, "debug_frame_start", 1)), 1)
        self.debug_frame_end = int(getattr(args, "debug_frame_end", 0))
        self.debug_track_ids = self._parse_debug_track_ids(getattr(args, "debug_track_ids", ""))
        self.debug_last_rows = []

    @staticmethod
    def _parse_debug_track_ids(value):
        if not value:
            return set()
        result = set()
        for part in str(value).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                result.add(int(part))
            except ValueError:
                continue
        return result

    def _debug_enabled_for_frame(self):
        if not self.debug_costs:
            return False
        if self.frame_id < self.debug_frame_start:
            return False
        if self.debug_frame_end > 0 and self.frame_id > self.debug_frame_end:
            return False
        return True

    @staticmethod
    def _resolve_ground_homography_path(args):
        homography_path = getattr(args, "ground_homography", "")
        if getattr(args, "disable_ground_motion", False):
            return ""
        if homography_path in {"", "none", "None", None}:
            return ""
        if homography_path != "auto":
            return homography_path

        video_path = osp.basename(str(getattr(args, "path", ""))).lower()
        calibration_dir = "Camera_calibration"
        if "center" in video_path:
            preferred = osp.join(calibration_dir, "H_center_to_ground_50pts.txt")
            fallback = osp.join(calibration_dir, "H_center_to_ground.txt")
            return preferred if osp.exists(preferred) else fallback
        if "left" in video_path:
            preferred = osp.join(calibration_dir, "H_left_to_ground_50pts.txt")
            fallback = osp.join(calibration_dir, "H_left_to_ground.txt")
            return preferred if osp.exists(preferred) else fallback
        return ""

    def _load_ground_homography(self, args):
        homography_path = self._resolve_ground_homography_path(args)
        if not homography_path:
            return None
        if not osp.exists(homography_path):
            raise FileNotFoundError(f"Ground homography file not found: {homography_path}")
        matrix = np.loadtxt(homography_path, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError(f"Ground homography must be 3x3: {homography_path}")
        print(f"Using ground homography for motion model: {homography_path}")
        return matrix

    def _project_bottom_centers_to_ground(self, tlbrs):
        if self.ground_homography is None or len(tlbrs) == 0:
            return [None] * len(tlbrs)

        tlbrs = np.asarray(tlbrs, dtype=float)
        bottom_centers = np.stack(
            [
                (tlbrs[:, 0] + tlbrs[:, 2]) * 0.5,
                tlbrs[:, 3],
                np.ones(len(tlbrs), dtype=float),
            ],
            axis=1,
        )
        projected = bottom_centers @ self.ground_homography.T
        denom = projected[:, 2:3]
        valid = np.abs(denom[:, 0]) > 1e-8
        ground_points = np.full((len(tlbrs), 2), np.nan, dtype=float)
        ground_points[valid] = projected[valid, :2] / denom[valid]
        return [point if np.all(np.isfinite(point)) else None for point in ground_points]

    def update(self, output_results, img_info, img_size,det_feats = None):
        self.frame_id += 1
        self.debug_last_rows = []
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh

        inds_low = scores > (self.args.track_thresh / 500)
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
    

        dets = bboxes[remain_inds]
        dets_second = bboxes[inds_second]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        det_feats_keep = det_feats[remain_inds] if det_feats is not None else None
        det_feats_second = det_feats[inds_second] if det_feats is not None else None        
        ground_points_keep = self._project_bottom_centers_to_ground(dets)
        ground_points_second = self._project_bottom_centers_to_ground(dets_second)

        if len(dets) > 0:
            '''Detections'''
            if det_feats_keep is None:
                detections = [
                    STrack(STrack.tlbr_to_tlwh(tlbr), s, motion_xy=motion_xy)
                    for tlbr, s, motion_xy in zip(dets, scores_keep, ground_points_keep)
                ]
            else:
                detections = [
                    STrack(STrack.tlbr_to_tlwh(tlbr), s, feat=f, motion_xy=motion_xy)
                    for tlbr, s, f, motion_xy in zip(dets, scores_keep, det_feats_keep, ground_points_keep)
                ]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        predicted_tracks = strack_pool
        primary_debug_records = [] if self._debug_enabled_for_frame() else None
        matches, u_track, u_detection = matching.associate_tracks_detections(
            predicted_tracks,
            detections,
            match_thresh=self.args.match_thresh,
            kf=self.kalman_filter,
            maha_thresh=getattr(self.args, "maha_thresh", None),
            maha_thresh_roi=getattr(self.args, "maha_thresh_roi", None),
            maha_roi_polygon=getattr(self.args, "maha_roi_polygon", None),
            use_maha_gate=getattr(self.args, "use_maha_gate", True),
            only_position=True,
            mot20=self.args.mot20,
            use_fuse_score_on_iou=True,
            frame_id=self.frame_id,
            velocity_min_speed=getattr(self.args, "velocity_min_speed", 1.0),
            debug_records=primary_debug_records,
            debug_stage="primary",
            debug_track_ids=self.debug_track_ids,
        )
        if primary_debug_records:
            self.debug_last_rows.extend(primary_debug_records)

        for itracked, idet in matches:
            track = predicted_tracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            if det_feats_second is None:
                detections_second = [
                    STrack(STrack.tlbr_to_tlwh(tlbr), s, motion_xy=motion_xy)
                    for tlbr, s, motion_xy in zip(dets_second, scores_second, ground_points_second)
                ]
            else:
                detections_second = [
                    STrack(STrack.tlbr_to_tlwh(tlbr), s, feat=f, motion_xy=motion_xy)
                    for tlbr, s, f, motion_xy in zip(dets_second, scores_second, det_feats_second, ground_points_second)
                ]
        else:
            detections_second = []
        r_tracked_stracks = [predicted_tracks[i] for i in u_track if predicted_tracks[i].state == TrackState.Tracked]
        match_thresh_low = max(self.args.match_thresh - 0.2, 0.0)
        secondary_debug_records = [] if self._debug_enabled_for_frame() else None
        matches, u_track, u_detection_second = matching.associate_tracks_detections(
            r_tracked_stracks,
            detections_second,
            match_thresh=match_thresh_low,
            kf=self.kalman_filter,
            maha_thresh=getattr(self.args, "maha_thresh", None),
            maha_thresh_roi=getattr(self.args, "maha_thresh_roi", None),
            maha_roi_polygon=getattr(self.args, "maha_roi_polygon", None),
            use_maha_gate=getattr(self.args, "use_maha_gate", True),
            only_position=True,
            mot20=self.args.mot20,
            use_fuse_score_on_iou=False,
            frame_id=self.frame_id,
            velocity_min_speed=getattr(self.args, "velocity_min_speed", 1.0),
            debug_records=secondary_debug_records,
            debug_stage="secondary",
            debug_track_ids=self.debug_track_ids,
        )
        if secondary_debug_records:
            self.debug_last_rows.extend(secondary_debug_records)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        unconfirmed_debug_records = [] if self._debug_enabled_for_frame() else None
        matches, u_unconfirmed, u_detection = matching.associate_tracks_detections(
            unconfirmed,
            detections,
            match_thresh=self.args.match_thresh,
            kf=self.kalman_filter,
            maha_thresh=getattr(self.args, "maha_thresh", None),
            maha_thresh_roi=getattr(self.args, "maha_thresh_roi", None),
            maha_roi_polygon=getattr(self.args, "maha_roi_polygon", None),
            use_maha_gate=getattr(self.args, "use_maha_gate", True),
            only_position=True,
            mot20=self.args.mot20,
            use_fuse_score_on_iou=True,
            frame_id=self.frame_id,
            velocity_min_speed=getattr(self.args, "velocity_min_speed", 1.0),
            debug_records=unconfirmed_debug_records,
            debug_stage="unconfirmed",
            debug_track_ids=self.debug_track_ids,
        )
        if unconfirmed_debug_records:
            self.debug_last_rows.extend(unconfirmed_debug_records)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
