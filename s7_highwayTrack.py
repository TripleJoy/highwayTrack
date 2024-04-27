import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from track.track_tools.TrackerArgs import *
from track.track_tools.highwayTracker import *
from track.track_tools.highwayTrack_tools import *
from det.yolov5.utils.datasets import LoadImages
from det.yolov5.utils.torch_utils import *
from det.det_tools.load_model import *
from det.det_tools.det_img_pre_process import *
from det.det_tools.det_and_nms import *
from det.det_tools.cfg_utils import *
from tools.tools_init import *
cv2.setNumThreads(-1)


@torch.no_grad()
def vehicle_highway_track(file_name, use_det_labels=True):
    logger = get_logger('vehicle-highway-track')
    logger.info('vehicle highway track start...')

    path_dict = get_path_dict(file_name)
    highwayTrack_labels_path = path_dict['highwayTrack-online']
    config = load_config(file_name)
    track_config = config['track_config']
    img_size = config['img_size']
    is_reverse = config['reverse']
    center_point = [img_size[0] // 2, img_size[1] // 2]
    track_args = TrackerArgs(track_thresh=track_config['track_thresh'],
                             track_buffer=track_config['track_buffer'],
                             match_thresh=track_config['match_thresh'],
                             aspect_ratio_thresh=track_config['aspect_ratio_thresh'],
                             min_box_area=track_config['min_box_area'],
                             img_height=img_size[0],
                             img_width=img_size[0],
                             mot20=track_config['mot20'])
    ori_pt, cal_params = read_cal_params(file_name, logger)
    hull = read_hull_points(file_name, logger)
    y_scope = [hull[0][0][1], hull[0][-1][1]]
    logger.info('processing...')
    pos_map = init_camera_to_world_map(img_size, cal_params, ori_pt, center_point)
    backSub = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    cap = get_video(file_name, logger)
    for f in tqdm(range(50), ncols=100):
        ret, frame = cap.read()
        if not ret:
            break

        if is_reverse:
            frame = cv2.flip(frame, 1)
        backSub.apply(frame)

    track_info = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    highwayTracker = HIGHWAYTracker(track_args, ori_pt, cal_params, center_point, y_scope, pos_map,img_size,
                                    frame_rate_default=track_config['frame_rate'])

    if use_det_labels:
        det_info, max_frame = read_det_info_in_lane_area(file_name, logger)

        t0 = time.time()
        for fid in tqdm(range(max_frame), ncols=100):
            ret, frame = cap.read()
            if not ret:
                break
            if is_reverse:
                frame = cv2.flip(frame, 1)
            fgMask = backSub.apply(frame)
            fgMask[fgMask == 127] = 0
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            mask = modify_mask(fgMask, [])
            if fid in det_info.keys():
                det = det_info[fid]
            else:
                det = []
            if is_reverse:
                for i in range(len(det)):
                    x1, y1, x2, y2, conf, cls = det[i]
                    x1 = img_size[0] - 1 - x1
                    x2 = img_size[0] - 1 - x2
                    x1,x2 = x2,x1
                    det[i] = [x1, y1, x2, y2, conf, cls]
            track_data,total_boxes = highwayTracker.update(det, mask)
            for t in track_data:
                x1, y1, x2, y2 = t.new_box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if is_reverse:
                    x1 = img_size[0] - 1 - x1
                    x2 = img_size[0] - 1 - x2
                    x1, x2 = x2, x1
                tid = t.track_id
                score = round(float(t.score), 3)
                cls = int(t.cls)
                speed = round(float(t.speed), 2)
                if speed < 1:
                    continue
                is_predict = int(t.is_predict)
                if tid not in track_info.keys():
                    track_info[tid] = []
                track_info[tid].append(
                    [fid+1,
                     x1, y1, x2, y2,
                     score, cls, speed, is_predict])
    else:
        weights = path_dict['det-model']
        source = path_dict['video']
        det_config = load_config(file_name)['det_config']
        imgsz = det_config['imgsz']
        device = det_config['device']
        conf_thres = det_config['conf_thres']
        iou_thres = det_config['iou_thres']
        max_det = det_config['max_det']
        classes = det_config['classes']
        cls_gap = 3
        for i in range(len(classes)):
            classes[i] += cls_gap
        cls_min_box_area = det_config['cls_min_box_area']
        half = det_config['half']

        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        model, names, imgsz, stride = load_model(weights, device, half, imgsz)
        # Dataloader
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True, left_padding=0)
        max_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()
        with tqdm(total=max_frame, ncols=100) as bar:
            for path, img, im0s, r_img0, vid_cap, s in dataset:
                p, im0, r_im0, fid = Path(path), im0s.copy(), r_img0.copy(), getattr(dataset, 'frame', 0)
                # print(img.shape,im0.shape,r_img0.shape)
                if fid >= max_frame:
                    break
                img = det_img_pre_process(device, half, img)
                det_, det_num = det_and_nms(model, img, conf_thres,
                                           iou_thres, classes, max_det,
                                           im0, cls_min_box_area, cls_gap)
                det = []
                for x1, y1, x2, y2, conf, cls in det_:
                    if y2 < y_scope[0]:
                        continue
                    if check_box_data([x1, y1, x2, y2], img_size, config['border_gap']):
                        det.append([x1, y1, x2, y2, conf, cls])
                ret, frame = cap.read()
                if not ret:
                    break
                if is_reverse:
                    frame = cv2.flip(frame, 1)
                fgMask = backSub.apply(frame)
                fgMask[fgMask == 127] = 0
                fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
                mask = modify_mask(fgMask, [])
                if is_reverse:
                    for i in range(len(det)):
                        x1, y1, x2, y2, conf, cls = det[i]
                        x1 = img_size[0] - 1 - x1
                        x2 = img_size[0] - 1 - x2
                        x1, x2 = x2, x1
                        det[i] = [x1, y1, x2, y2, conf, cls]
                track_data,total_boxes = highwayTracker.update(det, mask)
                for t in track_data:
                    x1, y1, x2, y2 = t.new_box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2),
                    tid = t.track_id
                    score = round(float(t.score), 3)
                    cls = int(t.cls)
                    speed = round(float(t.speed), 2)
                    if speed < 1:
                        continue
                    is_predict = int(t.is_predict)
                    if tid not in track_info.keys():
                        track_info[tid] = []
                    if is_reverse:
                        x1 = img_size[0] - 1 - x1
                        x2 = img_size[0] - 1 - x2
                        x1, x2 = x2, x1
                    track_info[tid].append(
                        [fid,
                         x1, y1, x2, y2,
                         score, cls, speed, is_predict])
                bar.update(1)
    track_data = []
    real_id = []
    id_now = 0
    for tid in track_info.keys():
        if tid not in real_id:
            real_id.append(tid)
            id_now += 1
        for t_info in track_info[tid]:
            t_info_ = [id_now] + t_info
            track_data.append(t_info_)
    t1 = time.time()
    logger.info(f'vehicle highway track finished, use-time={round(t1 - t0, 2)}s, vehicle-num={len(track_info.keys())}, '
                f'frames={max_frame}, time-per-frame={round((t1 - t0) / max_frame, 4)}s')
    save_highway_track_real_labels(highwayTrack_labels_path, track_data, logger)


if __name__ == '__main__':
    args = make_args()
    vehicle_highway_track(args.name,use_det_labels=args.use_det_labels)
