from tools.tools_init import *


def make_cal_img(file_name,logger,visualization=False):
    logger.info('make cal img start...')
    t0 = time.time()
    config = load_config(file_name)
    is_reverse = config['reverse']
    path_dict = get_path_dict(file_name)
    video_path = path_dict['video']
    cal_img_path = path_dict['cal-img']
    det_info, max_frame = read_det_info(file_name,logger)
    pts,_ = read_lane_area_mark_points(file_name,logger)

    cap = cv2.VideoCapture(video_path)
    flag, frame = cap.read()
    h,w,_ = frame.shape

    pts_tmp = []
    for pt in pts:
        pts_tmp.append([pt])
    pts_tmp = np.array(pts_tmp)
    hull = cv2.convexHull(pts_tmp)

    gap_frames = []
    for frame in range(max_frame):
        if frame not in det_info:
            gap_frames.append(frame)
    if len(gap_frames) == 0:
        logger.error('【ERROR】 No enough free lane img.')
        logger.error('【ERROR】 make cal img failed.')
        return
    sample_num = min(100,len(gap_frames) // 3 +1)
    gap_frames= random.sample(gap_frames, sample_num)
    img_queue = []
    img_gray_queue = []
    for f in gap_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        flag, frame = cap.read()
        if flag:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = get_mask_area_img(frame_gray,hull)
            img_queue.append(frame)
            img_gray_queue.append(frame_gray)
    cap.release()
    img_gray_queue = np.array(img_gray_queue)
    img_avg = img_gray_queue.mean(axis=0)
    img_avg = np.array(img_avg,dtype=np.uint8)
    avg = []
    for i in range(len(img_gray_queue)):
        img = np.array(img_gray_queue[i],dtype=np.int64)
        r = img - img_avg
        r = np.maximum(r, -r)
        loss = np.max(r)
        avg.append([i,loss])
    avg = sorted(avg, key=lambda cus: cus[1], reverse=False)
    save_img = img_queue[avg[0][0]]
    # if is_reverse:
    #     save_img = cv2.flip(save_img,1)
    t1 = time.time()
    logger.info(f'make cal img finished, use-time={round(t1 - t0, 2)}s')
    cv2.imwrite(cal_img_path,save_img)
    t2 = time.time()
    logger.info(f'===> save cal-img to \'{cal_img_path}\', use-time={round(t2 - t1, 3)}s')
    if visualization:
        p_name = f's2_cal_img.png'
        visualization_dir = path_dict['data-visualization-dir']
        cal_img_visualization_path = os.path.join(visualization_dir, p_name)
        cv2.imwrite(cal_img_visualization_path, save_img)
        logger.info(f'cal_img visualization ===> save to \'{cal_img_visualization_path}\'')


def make_lane_area_mark_points(file_name,logger):
    logger.info('make lane area mark points start...')
    t0 = time.time()
    path_dict = get_path_dict(file_name)
    config = load_config(file_name)
    is_reverse = config['reverse']
    lane_area_mark_points_path = path_dict['lane-area-mark-points']
    pre_track_labels_by_fid, pre_track_labels_by_vid, max_frame = read_pre_track_labels(file_name, logger)
    pts = [[],[]]
    for key in pre_track_labels_by_vid.keys():
        data = pre_track_labels_by_vid[key]
        _,_,_,_,y1,_,_ = data[0]
        _,_,_,_,y2,_,_ = data[-1]
        d = 0
        if y2 < y1:
            d = 1
        for t in data:
            frame, x1, y1, x2, y2, conf, cls = t
            if conf > 0.6:
                pts[d].append([x2,y2])
    sample_num = [min(200,len(pts[0])),min(200,len(pts[1]))]
    # lane_area_mark_points = [random.sample(pts[0], sample_num[0]), random.sample(pts[1], sample_num[1])]
    if not is_reverse:
        lane_area_mark_points = [random.sample(pts[0], sample_num[0]),random.sample(pts[1], sample_num[1])]
    else:
        lane_area_mark_points = [random.sample(pts[1], sample_num[1]), random.sample(pts[0], sample_num[0])]
    t1 = time.time()
    logger.info(f'make lane area mark points finished, use-time={round(t1 - t0, 2)}s')
    save_lane_area_mark_points(lane_area_mark_points_path,lane_area_mark_points, logger)
    return lane_area_mark_points


def pre_process(file_name,visualization=False):
    logger = get_logger('pre-process')
    lane_area_mark_points = make_lane_area_mark_points(file_name,logger)
    make_cal_img(file_name,logger,visualization=visualization)
    if visualization:
        path_dict = get_path_dict(file_name)
        config = load_config(file_name)
        is_reverse = config['reverse']
        p_name = f's2_lane_area_mark_points.png'
        img_path = path_dict['cal-img']
        img = cv2.imread(img_path)
        if is_reverse:
            img = cv2.flip(img, 1)
        visualization_dir = path_dict['data-visualization-dir']
        lane_area_mark_points_visualization_path = os.path.join(visualization_dir, p_name)
        draw_pts_on_img(img, lane_area_mark_points[0], cc=(0, 165, 255), point_weight=3)
        draw_pts_on_img(img, lane_area_mark_points[1], cc=(255, 255, 51), point_weight=3)
        if is_reverse:
            img = cv2.flip(img, 1)
        cv2.imwrite(lane_area_mark_points_visualization_path, img)
        logger.info(f'lane_area_mark_points visualization ===> save to \'{lane_area_mark_points_visualization_path}\'')


if __name__ == '__main__':
    args = make_args()
    pre_process(args.name,visualization=not args.non_visual)