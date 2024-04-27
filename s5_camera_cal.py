from camera_calibration.core_method import *
from camera_calibration.make_vp1 import *
from camera_calibration.xyz_trans import *
from tools.tools_init import *


def build_cal_points(file_name, logger, visualization=False):
    logger.info('build cal points start...')
    t0 = time.time()
    config = load_config(file_name)
    img_size = config['img_size']
    cal_config = config['camera_cal_config']
    max_dotted_line_num = cal_config['max_dotted_line_num']
    dotted_line_len = cal_config['dotted_line_len']
    dotted_line_gap_len = cal_config['dotted_line_gap_len']
    lane_width = cal_config['lane_width']
    lane_lines_info = read_lane_lines_info(file_name, logger)
    hull_points = read_hull_points(file_name, logger)
    cal_points = [[], []]
    dotted_lines_info = [[], []]
    path_dict = get_path_dict(file_name)
    img_path = path_dict['cal-img']
    img = cv2.imread(img_path)
    is_reverse = config['reverse']
    if is_reverse:
        img = cv2.flip(img, 1)
    cal_points_path = path_dict['cal-points']
    for d in [0, 1]:
        lane_lines = lane_lines_info[d]
        choose_hull = hull_points[d]
        for sub_lane_lines in lane_lines:
            if len(sub_lane_lines) > 1:
                dotted_line = sub_lane_lines[1:]
                dotted_line_new = []
                for dotted_sub_line in dotted_line:
                    (x1, y1), (x2, y2) = dotted_sub_line[0], dotted_sub_line[1]
                    dist = cv2.pointPolygonTest(np.array(choose_hull), (float(x1), float(y1)), True)
                    dist1 = cv2.pointPolygonTest(np.array(choose_hull), (float(x2), float(y2)), True)
                    if dist < 3 and dist1 < 3:
                        continue
                    dotted_line_new.append(dotted_sub_line)
                num = min(max_dotted_line_num,len(dotted_line_new))
                dotted_lines_info[d].append(dotted_line_new[:num])

    for d in [0, 1]:
        mark_index = [0]
        dotted_lines = dotted_lines_info[d]
        if len(dotted_lines) == 0:
            continue
        for i in range(1,len(dotted_lines)):
            mark_point = dotted_lines[i][0][1]
            tmp = []
            for idx,(_,p) in enumerate(dotted_lines[0]):
                k, _ = line_to_params([mark_point, p])
                if k is None:
                    k = 1000000
                tmp.append([idx,math.fabs(k)])
            tmp = sorted(tmp, key=lambda cus: cus[1], reverse=False)
            mark_index.append(tmp[0][0])
        for i in range(len(dotted_lines)):
            for j in range(len(dotted_lines[i])):
                sub_line = dotted_lines[i][j]
                x_pos = i * lane_width
                y_pos = (mark_index[i]+j) * (dotted_line_len + dotted_line_gap_len)
                cal_points[d].append([sub_line[1], [x_pos, y_pos]])
                cal_points[d].append([sub_line[0], [x_pos, y_pos + dotted_line_len]])
    t1 = time.time()
    logger.info(f'build cal points finished, use-time={round(t1 - t0, 3)}s')
    save_cal_points(cal_points_path, cal_points, logger)
    if visualization:
        visualization_dir = path_dict['data-visualization-dir']
        p_name = f's5_cal_points.png'
        cal_points_visualization_path = os.path.join(visualization_dir, p_name)
        bias = [7, 3]
        if is_reverse:
            img = cv2.flip(img, 1)
            bias = [-7, 3]
        for direction in [0, 1]:
            for pt, (x2, y2) in cal_points[direction]:
                pt = reg_pt(pt)
                if is_reverse:
                    pt[0] = img_size[0] - 1 - pt[0]
                pos = pt
                pos[0] += bias[0]
                pos[1] += bias[1]
                draw_text_on_img(img, f'({x2}, {y2})', pos, size=0.5, weight=2)
                draw_pts_on_img(img, [pos])

        cv2.imwrite(cal_points_visualization_path, img)
        logger.info(f'===> save to \'{cal_points_visualization_path}\'')


def get_vp1(file_name, logger, center_point):
    lane_lines = read_lane_lines_info(file_name, logger)
    input_x = []
    input_y = []
    for d in [0, 1]:
        for sub_lines in lane_lines[d]:
            if len(sub_lines) > 1:
            # if len(sub_lines) > 0:
                input_x.append(sub_lines[0])
                input_y.append(0)
    vp1_init = calc_cross_point(input_x[0], input_x[-1])
    if len(input_x) > 2:
        vp1_init = [[vp1_init[0]], [vp1_init[1]]]
        vp1, _ = get_vp1_by_lm([input_x, input_y], vp1_init)
    else:
        vp1 = vp1_init
    vp1 = [vp1[0] - center_point[0], vp1[1] - center_point[1]]
    return vp1


def init_params(file_name, logger, center_point):
    return [get_vp1(file_name, logger, center_point), 3000.0, 7]


def init_input(cal_points, center_point):
    input_x, input_y = [], []
    for d in [0, 1]:
        cal_points_d = cal_points[d]
        if len(cal_points_d) < 1:
            continue
        ori_c = cal_points_d[0][0]
        ori_c = [ori_c[0] - center_point[0], ori_c[1] - center_point[1]]
        for p_c, p_w in cal_points_d:
            p_c = [p_c[0] - center_point[0], p_c[1] - center_point[1]]
            input_x.append([p_c, p_w, ori_c])
            input_y.append(0)
    return [input_x, input_y]


def check_cal_points(file_name, logger):
    lane_lines_info = read_lane_lines_info(file_name, logger)
    cal_points = read_cal_points(file_name, logger)
    flag = True
    count = [0,0]
    for d in [0, 1]:
        for sub_lane_lines in lane_lines_info[d]:
            if len(sub_lane_lines) > 1:
                count[d] += 1
    if count[0] <=1 and count[1] <= 1:
        flag = False
    if len(cal_points[0]) < 6 and len(cal_points[1]) < 6:
        flag =False
    if not flag:
        logger.info("【ERROR】No enough cal points for camera cal.")
    return flag


def camera_calibration(file_name, logger):
    logger.info('camera calibration start...')
    t0 = time.time()
    path_dict = get_path_dict(file_name)
    cal_params_path = path_dict['cal-params']
    config = load_config(file_name)
    is_reverse = config['reverse']
    img_size = config['img_size']
    center_point = [img_size[0] / 2, img_size[1] / 2]
    if not check_cal_points(file_name, logger):
        return
    cal_points = read_cal_points(file_name, logger)
    input_ = init_input(cal_points, center_point)
    params_init = init_params(file_name, logger, center_point)
    params, step, mse = get_cal_params_by_lm(input_, params_init)
    logger.info("step=%d, mse=%.8f" % (step, abs(mse)))
    f, h = params
    vp1, _, _ = params_init
    phi, theta = get_phi_and_theta(f, vp1)
    if is_reverse:
        if len(cal_points[1]) > 0:
            ori_point = cal_points[1][0][0]
        else:
            ori_point = cal_points[0][0][0]
    else:
        if len(cal_points[0]) > 0:
            ori_point = cal_points[0][0][0]
        else:
            ori_point = cal_points[1][0][0]
    vp1 = [vp1[0] + center_point[0], vp1[1] + center_point[1]]
    logger.info("ori_pt=[%.1f, %.1f],  vp1=[%.1f, %.1f],  f=%.2f,  h=%.2f"
                % (ori_point[0],ori_point[1],vp1[0], vp1[1], f, h))
    logger.info("phi=%.2f,  theta=%.2f" %
                (phi, theta))
    t1 = time.time()
    logger.info(f'camera calibration finished, use-time={round(t1 - t0, 2)}s')

    cal_params = [ori_point, vp1, f, h]
    save_cal_params(cal_params_path, cal_params, logger)


def camera_cal(file_name,visualization=False):
    logger = get_logger('camera-cal')
    build_cal_points(file_name,logger,visualization=visualization)
    camera_calibration(file_name, logger)


if __name__ == '__main__':
    args = make_args()
    camera_cal(args.name,visualization=not args.non_visual)
