import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
from tools.tools_init import *
from camera_calibration.make_vp1 import *


def remake_lane_area_mark_points(data):
    input_point = []
    input_label = []
    for f, x, y in data:
        f, x, y = int(f), int(x), int(y)
        input_point.append([x, y])
        input_label.append(f)
    return np.array(input_point), np.array(input_label)


def get_hull(predictor, input_point, input_label):
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
    masks, s, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )
    mask = masks[0]
    score = s[0]
    h, w = mask.shape[-2:]
    m = mask.reshape(h, w) * 255
    m = np.array(m, dtype=np.uint8)
    boundary_points, hull = find_max_region(m)
    hull = np.array(hull, np.int32)
    return boundary_points, hull


def make_hull_lines_new(hull,vp,img_size, img):

    w,h = img_size
    bottom_pos,right_pos = 0,0
    for pt in hull:
        bottom_pos = max(bottom_pos,pt[1])
        right_pos = max(right_pos,pt[0])
    limit_bottom = bottom_pos - (bottom_pos-vp[1]) / 10
    limit_right = right_pos - (right_pos-vp[0]) / 10
    l1 = [[0,limit_bottom],[1920,limit_bottom]]
    l2 = [[limit_right,0],[limit_right,1080]]
    draw_lines_on_img(img,[l1,l2],lc=get_color(2))
    choose_points = []
    un_choose_points = []
    for pt in hull:
        if pt[0] >= limit_right or pt[1]>=limit_bottom:
            choose_points.append(pt)
        else:
            un_choose_points.append(pt)
    # draw_pts_on_img(img,choose_point)
    # draw_pts_on_img(img,[vp],cc=get_color(1))
    # show_img(img)
    lines_info = []
    for pt in choose_points:
        line = [vp,pt]
        line_bottom = [[0,h],[w,h]]
        pt_cross = calc_cross_point(line,line_bottom)
        if pt_cross is not None:
            x,y = pt_cross
            lines_info.append([line,x])
    lines_info = sorted(lines_info, key=lambda cus: cus[1], reverse=False)
    l1 = lines_info[0][0]
    l2 = lines_info[-1][0]
    k1,b1 = line_to_params(l1)
    k2,b2 = line_to_params(l2)
    return [[k1,b1],[k2,b2]],choose_points,un_choose_points


def make_hull_lines(choose_hull, img_size):
    h, w = img_size
    total_dis = 0
    hull_lines = []
    for i in range(len(choose_hull) - 1):
        hull_lines.append([choose_hull[i], choose_hull[i + 1]])
    hull_lines.append([choose_hull[len(choose_hull) - 1], choose_hull[0]])
    for ls in hull_lines:
        total_dis += line_dis(ls)
    total_points = 500
    gap_dis = total_dis / total_points
    pts = []
    for ls in hull_lines:
        divide_points = divide_line_to_points(ls, gap_dis)
        d_pts = []
        for pt in divide_points:
            if 10 < pt[0] < w - 10:
                d_pts.append(pt)
        pts += d_pts
    out_points = pts.copy()
    temp_points = []
    for i in range(3):
        choose_points, in_points, out_points = ransac(out_points)
        y_mid = 0
        for pt in in_points:
            y_mid += pt[1]
        y_mid /= len(in_points)
        temp_points.append([in_points, y_mid])
        if len(out_points) < len(pts) / 6:
            break
    if len(temp_points) < 2:
        return False,[]
    temp_points = sorted(temp_points, key=lambda cus: cus[1], reverse=False)
    hull_lines = []
    for i in range(2):
        a, b = linear_regression(temp_points[i][0])
        hull_lines.append([a, b])
    return True,hull_lines


def make_hull_scope(hull_lines, img_size):
    h, w = img_size
    l1, l2 = hull_lines
    k1, b1 = l1
    k2, b2 = l2
    l1 = params_to_line(k1, b1, w)
    l2 = params_to_line(k2, b2, w)
    c_point = calc_cross_point(l1, l2)
    # y_scope = [c_point[1] + (h - c_point[1]) / 4, h]
    y_scope = [c_point[1] + (h - c_point[1]) / 6, h]
    return y_scope


def remake_hull_points(hull_lines, y_scope):
    hulls = []
    a1, b1 = hull_lines[0]
    a2, b2 = hull_lines[1]
    y1, y2 = y_scope
    hulls.append([(y1 - b1) / a1, y1])
    hulls.append([(y1 - b2) / a2, y1])
    hulls.append([(y2 - b2) / a2, y2])
    hulls.append([(y2 - b1) / a1, y2])
    if hulls[0][0] > hulls[1][0]:
        hulls[0], hulls[1] = hulls[1], hulls[0]
        hulls[2], hulls[3] = hulls[3], hulls[2]
    return reg_pts(hulls)


def calc_hull_points(file_name, logger, predictor, input_point, input_label, img_size, direction, img, y_scope=None):
    boundary_points, hull = get_hull(predictor, input_point, input_label)
    traj_lines = get_traj_lines(file_name, logger, hull, direction)
    vp = make_vp1_by_traj(traj_lines)
    # draw_lines_on_img(img,traj_lines)
    # draw_pts_on_img(img,[vp],cc=get_color(2))
    # show_img(img)
    hull_lines, choose_points,un_choose_points = make_hull_lines_new(hull, vp, img_size, img.copy())
    # flag,hull_lines = make_hull_lines(hull, img_size)
    # count = 0
    # while not flag:
    #     count+=1
    #     logger.info(f'ransac round {count}..')
    #     flag, hull_lines = make_hull_lines(hull, img_size)
    #     if count >= 10:
    #         logger.error(f'【ERROR】build hull points failed..')
    #         return [],None
    if y_scope is None:
        y_scope = make_hull_scope(hull_lines, img_size)
    hull_new = np.array(remake_hull_points(hull_lines, y_scope), np.int32)
    return boundary_points, hull, hull_new, y_scope, vp, choose_points,un_choose_points


def init_data(file_name,logger):
    path_dict = get_path_dict(file_name)
    img_path = path_dict['cal-img']
    hull_points_path = path_dict['hull-points']
    config = load_config(file_name)
    seg_model_path = path_dict['seg-model']
    seg_config = config['area_division']

    image = cv2.imread(img_path)
    is_reverse = config['reverse']
    if is_reverse:
        image = cv2.flip(image, 1)
    im0 = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    image_size = [h, w]

    input_point, input_label = read_lane_area_mark_points(file_name,logger)
    input_point = np.array(input_point)
    input_label = np.array(input_label)
    sam_checkpoint = seg_model_path
    device = 'cuda'
    sam = sam_model_registry[seg_config['model_type']](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    return hull_points_path, predictor, input_point, input_label, image, im0, image_size


def lane_area_division(file_name,visualization=False):
    logger = get_logger('lane-area-division')
    t0 = time.time()
    hull_points_path, predictor, input_point, input_label, image, im0, image_size = init_data(file_name,logger)
    t1 = time.time()
    logger.info(f'load seg model finished, use-time={round(t1 - t0, 2)}s')
    logger.info('lane area division start...')
    input_label = 1 - input_label
    boundary_points_left, hull_left, hull_new_left, y_scope, vp_left, choose_points_left, un_choose_points_left = \
        calc_hull_points(file_name, logger, predictor, input_point, input_label, image_size,0,image)
    input_label = 1 - input_label
    boundary_points_right, hull_right, hull_new_right, _, vp_right, choose_points_right, un_choose_points_right= \
        calc_hull_points(file_name, logger, predictor, input_point, input_label, image_size,1,image,y_scope=y_scope)
    hull_points = [hull_new_left,hull_new_right]
    t2 = time.time()
    logger.info(f'lane area division finished, use-time={round(t2 - t1, 2)}s')
    save_hull_points(hull_points_path,hull_points, logger)
    if visualization:
        path_dict = get_path_dict(file_name)
        img_path = path_dict['cal-img']
        img = cv2.imread(img_path)
        config = load_config(file_name)
        is_reverse = config['reverse']
        if is_reverse:
            img = cv2.flip(img, 1)
        visualization_dir = path_dict['data-visualization-dir']
        lane_area_division_visualization_0_path = os.path.join(visualization_dir, 's3_lane_area_division_0.png')
        img_ = img.copy()
        draw_pts_on_img(img_,boundary_points_left)
        draw_pts_on_img(img_,boundary_points_right)
        if is_reverse:
            img_ = cv2.flip(img_, 1)
        cv2.imwrite(lane_area_division_visualization_0_path, img_)
        logger.info(f'lane_area_division visualization ===> save to \'{lane_area_division_visualization_0_path}\'')
        lane_area_division_visualization_1_path = os.path.join(visualization_dir, 's3_lane_area_division_1.png')
        img_ = img.copy()
        draw_area_on_img(img_,hull_left,weight=3)
        draw_pts_on_img(img_,hull_left)
        draw_area_on_img(img_,hull_right,weight=3)
        draw_pts_on_img(img_, hull_right)
        if is_reverse:
            img_ = cv2.flip(img_, 1)
        cv2.imwrite(lane_area_division_visualization_1_path, img_)
        logger.info(f'lane_area_division visualization ===> save to \'{lane_area_division_visualization_1_path}\'')
        lane_area_division_visualization_2_path = os.path.join(visualization_dir, 's3_lane_area_division_2.png')
        img_ = img.copy()
        draw_pts_on_img(img_,un_choose_points_left,cc=get_color(0))
        draw_pts_on_img(img_,choose_points_left,cc=get_color(1))
        draw_pts_on_img(img_, un_choose_points_right,cc=get_color(0))
        draw_pts_on_img(img_, choose_points_right,cc=get_color(1))
        draw_pts_on_img(img_,[vp_left,vp_right],get_color(2))
        if is_reverse:
            img_ = cv2.flip(img_, 1)
        cv2.imwrite(lane_area_division_visualization_2_path, img_)
        logger.info(f'lane_area_division visualization ===> save to \'{lane_area_division_visualization_2_path}\'')
        lane_area_division_visualization_3_path = os.path.join(visualization_dir, 's3_lane_area_division_3.png')
        img_ = img.copy()
        draw_area_on_img(img_,hull_new_left,weight=3)
        draw_area_on_img(img_,hull_new_right,weight=3)
        draw_pts_on_img(img_,hull_new_left)
        draw_pts_on_img(img_,hull_new_right)
        if is_reverse:
            img_ = cv2.flip(img_, 1)
        cv2.imwrite(lane_area_division_visualization_3_path, img_)
        logger.info(f'lane_area_division visualization ===> save to \'{lane_area_division_visualization_3_path}\'')


def make_vp1_by_traj(traj_lines):
    input_x = traj_lines
    input_y = np.zeros(len(traj_lines))
    vp1_init = calc_cross_point(input_x[0], input_x[-1])
    vp1_init = [[vp1_init[0]], [vp1_init[1]]]
    vp1, _ = get_vp1_by_lm([input_x, input_y], vp1_init)
    return vp1


def get_traj_lines(file_name,logger,hull,direction):
    pre_track_labels_by_fid, pre_track_labels_by_vid, max_frame = read_pre_track_labels(file_name, logger)
    config = load_config(file_name)
    is_reverse = config['reverse']
    traj_lines = []
    for key in pre_track_labels_by_vid.keys():
        data = pre_track_labels_by_vid[key]
        if len(data) <= 10:
            continue
        _, _, _, _, y1, _, _ = data[0]
        _, _, _, _, y2, _, _ = data[-1]
        d = 0
        if y2 < y1:
            d = 1
        if is_reverse:
            d = 1-d
        if d != direction:
            continue
        pts = []
        conf_avg = 0
        for t in data:
            frame, x1, y1, x2, y2, conf, cls = t
            # p = [(x1 + x2) / 2, y2]
            p = [x2, y2]
            if cv2.pointPolygonTest(hull, p, False) < 0:
                # draw_box_on_img(img,[x1,y1,x2,y2])
                # show_img(img)
                continue
            pts.append(p)
            conf_avg+=conf
        if len(pts) <=10:
            continue
        conf_avg/=len(data)
        if conf_avg <=0.5:
            continue
        k_, b_ = linear_regression(pts)
        line = params_to_line(k_,b_)
        traj_lines.append(line)
    return traj_lines


if __name__ == '__main__':
    args = make_args()
    lane_area_division(args.name,visualization=not args.non_visual)
