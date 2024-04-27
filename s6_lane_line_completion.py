from camera_calibration.xyz_trans import *
from tools.tools_init import *


def lane_line_completion(file_name,visualization=False):
    logger = get_logger('lane-line-completion')
    config = load_config(file_name)
    lane_line_mark_points = config['lane_line_completion']
    if lane_line_mark_points is None:
        logger.info('No lane-line-mark-points for lane line completion.')
        return
    path_dict = get_path_dict(file_name)
    img_size = config['img_size']
    center_point = [img_size[0] // 2, img_size[1] // 2]
    img_path = path_dict['cal-img']
    img = cv2.imread(img_path)
    is_reverse = config['reverse']
    if is_reverse:
        img = cv2.flip(img, 1)
    ori_pt, cal_params = read_cal_params(file_name, logger)
    lane_lines = read_lane_lines_info(file_name,logger)
    hull_points = read_hull_points(file_name,logger)
    hull_left = hull_points[0]
    l1 = [hull_left[0],hull_left[1]]
    l2 = [hull_left[2],hull_left[3]]

    lane_lines_real = []
    for x in lane_line_mark_points:
        p1 = world2_xyz_to_camera([x, 0, 0], cal_params, ori_pt, center_point)
        p2 = world2_xyz_to_camera([x, 40, 0], cal_params, ori_pt, center_point)
        ll = [p1, p2]
        cp1 = calc_cross_point(l1, ll)
        cp2 = calc_cross_point(l2, ll)
        lane_lines_real.append([[cp1, cp2]])
    for d in [0,1]:
        for i in range(len(lane_lines[d])):
            sub_lines = lane_lines[d][i]
            if i == 0 and len(sub_lines) == 1:
                continue
            p1, p2 = sub_lines[0]
            p1_w = camera_to_world2_xy(p1, cal_params, ori_pt, center_point=center_point)
            p2_w = camera_to_world2_xy(p2, cal_params, ori_pt, center_point=center_point)
            real_x = (p1_w[0] + p2_w[0]) / 2

            idx = 0
            eps = 100000
            for i in range(len(lane_line_mark_points)):
                if eps > math.fabs(real_x - lane_line_mark_points[i]):
                    eps = math.fabs(real_x - lane_line_mark_points[i])
                    idx = i
            if len(sub_lines) > 1:
                lane_lines_real[idx] = sub_lines[1:]
            else:
                lane_lines_real[idx] = [sub_lines[0]]
    lane_lines_completion_path = path_dict['lane-lines-completion']
    save_lane_lines_completion(lane_lines_completion_path, lane_lines_real, logger)
    if visualization:
        for lines in lane_lines_real:
            draw_lines_on_img(img, lines, lc=get_color(1), show_endpoint=True)
        visualization_dir = path_dict['data-visualization-dir']
        p_name = f's6_lane_line_completion.png'
        lane_line_completion_visualization_path = os.path.join(visualization_dir, p_name)
        if is_reverse:
            img = cv2.flip(img, 1)
        cv2.imwrite(lane_line_completion_visualization_path, img)
        logger.info(f'===> save to \'{lane_line_completion_visualization_path}\'')


if __name__ == '__main__':
    args = make_args()
    lane_line_completion(args.name,visualization=not args.non_visual)
