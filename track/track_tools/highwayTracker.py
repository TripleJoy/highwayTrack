from track.ByteTrack.yolox.tracker.kalman_filter import KalmanFilter
from track.ByteTrack.yolox.tracker import matching
from track.ByteTrack.yolox.tracker.basetrack import BaseTrack, TrackState
from track.track_tools.highwayTrack_tools import *


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, det, ori_pt, cal_params, center_point, y_scope, frame_rate_default=25):
        self.ori_pt = ori_pt
        self.cal_params = cal_params
        self.center_point = center_point
        self.y_scope = y_scope
        self.frame_rate_default = frame_rate_default
        self.cls_tmp = [0, 0, 0]
        self.k, self.b = None, None

        x1, y1, x2, y2, score, cls = det
        self.det = det
        self.box = np.array([x1, y1, x2, y2], dtype=np.float)
        self.new_box = self.box
        self._tlwh = np.asarray(self.tlbr_to_tlwh(self.box), dtype=np.float)

        self.vehicle_sizes = []
        self.boxes = [self.box]
        self.predict_boxes = []
        self.vehicle_positions = []
        self.last_positions = None
        self.score = score
        self.scores = [self.score]
        self.tracklet_len = 0
        self.cls_tmp[int(cls)] += 1

        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.is_predict = False

    @staticmethod
    def calc_box_occupy(box1, box2):
        overlap_x1 = max(box1[0], box2[0])
        overlap_y1 = max(box1[1], box2[1])
        overlap_x2 = min(box1[2], box2[2])
        overlap_y2 = min(box1[3], box2[3])

        # 计算重叠部分的宽度和高度
        overlap_width = overlap_x2 - overlap_x1
        overlap_height = overlap_y2 - overlap_y1

        # 如果宽度和高度都大于0，则计算面积；否则，面积为0
        if overlap_width > 0 and overlap_height > 0:
            area_both = overlap_width * overlap_height
        else:
            return 10
        x1, y1, x2, y2 = box1
        x1_, y1_, x2_, y2_ = box2
        area_a = (y2 - y1) * (x2 - x1)
        area_b = (y2_ - y1_) * (x2_ - x1_)
        return area_a / area_b

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            # print('mean------------\n', multi_mean.shape)
            # print('mean-val:   ', multi_mean)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def predict_next_point(points):
        # 生成带有 id 的数据列表
        data_with_id = [(i + 1, point) for i, point in enumerate(points)]

        # 提取 ids, x 坐标和 y 坐标
        ids = np.array([item[0] for item in data_with_id])
        x_coords = np.array([item[1][0] for item in data_with_id])
        y_coords = np.array([item[1][1] for item in data_with_id])

        # 对 x 和 y 坐标进行线性拟合
        poly_x = np.poly1d(np.polyfit(ids, x_coords, 1))
        poly_y = np.poly1d(np.polyfit(ids, y_coords, 1))

        pre_id = len(points) + 1
        x_pred = poly_x(pre_id)
        y_pred = poly_y(pre_id)

        return [x_pred, y_pred]

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.tracklet_len = 1

        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.is_predict = False

    def re_activate(self, new_track, frame_id, new_id=False):
        if self.tracklet_len >= 4:
            if self.calc_box_occupy(new_track.box, self.predict_box) > 1.2:
                self.predict_update(frame_id)
                return
        self.box = new_track.box
        self.score = new_track.score
        self.scores += [self.score]
        for i in range(len(self.cls_tmp)):
            self.cls_tmp[i] += new_track.cls_tmp[i]
        self.tracklet_len += 1
        self.frame_id = frame_id

        self.state = TrackState.Tracked
        self.is_activated = True
        if new_id:
            self.track_id = self.next_id()
        self.is_predict = False

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        if self.tracklet_len >= 4:
            if self.calc_box_occupy(new_track.box, self.predict_box) > 1.2:
                self.predict_update(frame_id)
                return
        self.box = new_track.box
        self.score = new_track.score
        self.scores += [self.score]
        for i in range(len(self.cls_tmp)):
            self.cls_tmp[i] += new_track.cls_tmp[i]
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.state = TrackState.Tracked
        self.is_activated = True
        self.is_predict = False

    def predict_update(self, frame_id):
        self.frame_id = frame_id
        predict_box = self.predict_box
        if predict_box[3] >= self.y_scope[1] or predict_box[3] <= self.y_scope[0]:
            return False
        self.box = predict_box
        self.score = -1
        self.tracklet_len += 1

        self.state = TrackState.Tracked
        self.is_activated = True
        self.is_predict = True
        return True

    def fix_update(self, boxes, mask, pos_map,img_size):
        box = self.box
        box = fix_bounding_box(box, boxes, mask, img_size=img_size)
        flag1,data = \
            get_vehicle_infos(box, self.cls,mask, self.cal_params, self.ori_pt, self.center_point, pos_map)
        if not flag1:
            # print(self.is_predict)
            self.is_activated = False
            self.state = TrackState.Removed
            return
        vehicle_length, vehicle_width, vehicle_pos_xy, box = data
        self._tlwh = np.asarray(self.tlbr_to_tlwh(box), dtype=np.float)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(self._tlwh))
        self.new_box = box
        self.boxes.append(self.new_box)
        self.predict_boxes.append(self.predict_box)
        self.last_positions = vehicle_pos_xy
        self.vehicle_positions.append(vehicle_pos_xy)
        self.vehicle_sizes.append([vehicle_length, vehicle_width])

    @property
    # @jit(nopython=True)
    def predict_box(self):
        if self.tracklet_len <= 4:
            # print('predict_box: ',self.track_id, 0)
            return self.tlwh_to_tlbr(self.tlwh)
        else:
            position_predict_x, position_predict_y = self.position_predict
            vehicle_length, vehicle_width = self.vehicle_size
            p1 = [position_predict_x + vehicle_width / 2, position_predict_y, 0]
            p2 = [position_predict_x - vehicle_width / 2, position_predict_y, 0]
            p3 = [position_predict_x - vehicle_width / 2, position_predict_y + vehicle_length, 0]
            right, _ = world2_xyz_to_camera(p1, self.cal_params, self.ori_pt, self.center_point)
            _, bottom = world2_xyz_to_camera(p2, self.cal_params, self.ori_pt, self.center_point)
            left, _ = world2_xyz_to_camera(p3, self.cal_params, self.ori_pt, self.center_point)
            # height = self.tlwh[3]
            ratio_avg = 0
            boxes = self.boxes
            boxes = boxes[-min(5, len(boxes)):]
            for x1, y1, x2, y2 in boxes:
                w = x2 - x1
                h = y2 - y1
                ratio_avg += h / w
            ratio_avg /= len(boxes)
            height = (right - left) * ratio_avg
            top = bottom - height
            predict_box_ = np.asarray([left, top, right, bottom], dtype=np.float)
            box = numpy_to_int_list(predict_box_)
            return predict_box_

    @property
    # @jit(nopython=True)
    def speed(self):
        if self.tracklet_len <= 1:
            return -1
        else:
            start_pos = self.vehicle_positions[0]
            end_pos = self.vehicle_positions[-1]
            during_frames = self.frame_id - self.start_frame
            dis = dis_between_two_points(start_pos, end_pos)
            speed = dis / during_frames * self.frame_rate_default * 3600 / 1000.0
            return speed

    @property
    # @jit(nopython=True)
    def direction(self):
        if self.tracklet_len <= 1:
            return None
        else:
            x_start, y_start = self.vehicle_positions[0]
            x_end, y_end = self.last_positions
            if y_end > y_start:
                direction = 0
            else:
                direction = 1
            return direction

    @property
    # @jit(nopython=True)
    def position_predict(self):
        if self.tracklet_len <= 3:
            return None
        else:

            return self.predict_next_point(self.vehicle_positions)

    @property
    # @jit(nopython=True)
    def vehicle_size(self):
        v_l, v_w = 0, 0
        frame_nums = min(10,len(self.vehicle_sizes))
        for vehicle_len, vehicle_width in self.vehicle_sizes[-frame_nums:]:
            v_l += vehicle_len
            v_w += vehicle_width
        return [v_l / frame_nums, v_w / frame_nums]

    @property
    # @jit(nopython=True)
    def score_avg(self):
        return sum(self.scores) / len(self.scores)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        # ret = self.tlwh.copy()
        # ret[2:] += ret[:2]
        ret = self.predict_box.copy()
        return ret

    @property
    # @jit(nopython=True)
    def cls(self):
        max_cls = max(self.cls_tmp)
        cls_ = self.cls_tmp.index(max_cls)
        return cls_

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
        return self.tlwh_to_xyah(self.tlwh)

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


class HIGHWAYTracker(object):
    def __init__(self, args, ori_pt, cal_params, center_point, y_scope, pos_map, img_size, frame_rate_default=25):
        self.img_size = img_size
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        # self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.ori_pt = ori_pt
        self.cal_params = cal_params
        self.center_point = center_point
        self.y_scope = y_scope
        self.pos_map = pos_map
        self.frame_rate_default = frame_rate_default
        self.buffer_size = int(frame_rate_default / 25.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, dets, mask):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(dets) == 0:
            dets_keep = []
            dets_second = []
        else:
            dets_keep = []
            dets_second = []
            for det in dets:
                x1, y1, x2, y2, score_, cls = det
                if score_ > self.args.track_thresh:
                    dets_keep.append(det)
                else:
                    if score_ > 0.1:
                        dets_second.append(det)
        # total_boxes = [det[:4] for det in dets]
        if len(dets_keep) > 0:
            '''Detections'''
            detections_keep = [STrack(det, self.ori_pt, self.cal_params, self.center_point, self.y_scope,
                                      frame_rate_default=self.frame_rate_default) for
                               det in dets_keep]
        else:
            detections_keep = []

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
        dists = matching.iou_distance(strack_pool, detections_keep)
        dists = matching.fuse_score(dists, detections_keep)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_keep[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(det, self.ori_pt, self.cal_params, self.center_point, self.y_scope,
                                        frame_rate_default=self.frame_rate_default) for det in dets_second]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
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
                if track.tracklet_len <= 3:
                    track.mark_lost()
                    lost_stracks.append(track)
                else:
                    flag = track.predict_update(self.frame_id)
                    if flag:
                        activated_starcks.append(track)
                    else:
                        track.mark_removed()
                        removed_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections_keep[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
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
        total_boxes = [t.box for t in output_stracks]
        for t in output_stracks:
            t.fix_update(total_boxes,mask,self.pos_map,self.img_size)
        output_stracks = [track for track in output_stracks if track.is_activated]
        self.removed_stracks.extend([track for track in output_stracks if not track.is_activated])
        return output_stracks,total_boxes


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


