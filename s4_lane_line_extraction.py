import sys

sys.path.append("..")
from tools.tools_init import *


def get_points_from_arr(img0, extra_boxes=None):
    pts = []
    row, col = np.nonzero(img0)
    for y, x in zip(row, col):
        flag = 1
        if extra_boxes is not None:
            for extra_box in extra_boxes:
                x1, y1, x2, y2 = extra_box
                if x1 <= x <= x2 and y1 <= y <= y2:
                    flag = 0
                    break
        if flag:
            pts.append([x, y])
    return pts


def check_area_points(pts, choose_hull, border_gap_default=2):
    pts_new = []
    for pt in pts:
        x, y = pt
        # print(x,y)
        dist = cv2.pointPolygonTest(np.array(choose_hull), (float(x), float(y)), True)
        if dist < border_gap_default:
            continue
        pts_new.append(pt)
    return pts_new


def divide_pts(pts, choose_hull,visualization=False):
    w_with_default = 1
    h, w = 200, 100
    # h, w = 200, 200
    pts_min = h // 3
    scale = 3
    resize = (w * scale, h * scale)
    dst_ = [[0, 0], [w, 0], [w, h], [0, h]]
    bg_img = np.zeros((h, w))
    perspective_mat = get_perspective_mat(choose_hull, dst_)
    pts_new = []
    for pt in pts:
        pt_new = point_perspective_trans(pt, perspective_mat)
        pts_new.append(pt_new)
        x, y = pt_new
        x = math.floor(x)
        y = math.floor(y)
        if x <= 0 or x >= w or y <= 0 or y >= h:
            continue
        bg_img[y][x] = 1
    kernel = np.ones((3, 3), np.uint8)
    bg_img_1 = cv2.dilate(bg_img, kernel, iterations=1)
    pts_num = np.sum(bg_img_1 == 1)
    w_index = np.sum(bg_img_1, axis=0)
    pts_exist_line_num = np.sum(w_index >= 5)
    pts_exist_line_num = max(int(pts_exist_line_num), 1)
    avg_pts_num = min(pts_num // pts_exist_line_num + 1, pts_min)
    for i in range(len(w_index)):
        if w_index[i] < avg_pts_num:
            bg_img_1[0:h, i:i + 1] = 0
    kernel = np.ones((3, 3), np.uint8)
    bg_img_2 = cv2.dilate(bg_img_1, kernel, iterations=1)
    w_index = np.sum(bg_img_2, axis=0)
    w_blank = []
    for i in range(len(w_index)):
        if w_index[i] == 0:
            w_blank.append(i)
    w_blank.append(w + 1)
    w_gap = []
    w_start = 0
    for i in range(1, len(w_blank)):
        if w_blank[i] != w_blank[i - 1] + 1:
            if w_blank[i - 1] - w_blank[w_start] + 1 >= w_with_default:
                w_gap.append([w_blank[w_start], w_blank[i - 1], w_blank[i - 1] - w_blank[w_start]])
                w_start = i
    w_mid = []
    for w_data in w_gap:
        # print(w_data)
        w_mid.append((w_data[1] + w_data[0]) // 2)
    pts_div = [[] for i in range(len(w_mid) + 1)]
    pts_new_div = [[] for i in range(len(w_mid) + 1)]
    for i in range(len(pts_new)):
        pt = pts[i]
        pt_new = pts_new[i]
        x, y = pt_new
        if w_index[int(x)] == 0:
            continue
        flag = 1
        for j in range(len(w_mid)):
            limit = w_mid[j]
            if pt_new[0] <= limit:
                flag = 0
                pts_div[j].append(pt)
                pts_new_div[j].append(pt_new)
                break
        if flag:
            pts_div[len(w_mid)].append(pt)
            pts_new_div[len(w_mid)].append(pt_new)
    img_list = []
    if visualization:
        bg_img = np.array([255*p for p in bg_img],dtype=np.uint8)
        bg_img_1 = np.array([255*p for p in bg_img_1],dtype=np.uint8)
        bg_img_2 = np.array([255*p for p in bg_img_2],dtype=np.uint8)
        bg_img = cv2.resize(bg_img, resize, interpolation=cv2.INTER_LINEAR)
        bg_img_1 = cv2.resize(bg_img_1, resize, interpolation=cv2.INTER_LINEAR)
        bg_img_2 = cv2.resize(bg_img_2, resize, interpolation=cv2.INTER_LINEAR)
        bg_img_3 = np.zeros((h, w, 3))
        show_lines = []
        for xx in w_mid:
            show_lines.append([[xx, 0], [xx, h]])
        for pts_d in pts_new_div:
            draw_lines_on_img(bg_img_3, show_lines, line_weight=1, lc=(255, 255, 255))
            draw_pts_on_img(bg_img_3, pts_d, point_weight=1, cc=(255, 255, 255))
        bg_img_3 = cv2.resize(bg_img_3, resize, interpolation=cv2.INTER_LINEAR)
        img_list = [bg_img, bg_img_1, bg_img_2, bg_img_3]
    return pts_div, img_list


def get_hull_data(hull_points):
    hull_left, hull_right = [], []
    for f, x, y in hull_points:
        if f == 0:
            hull_left.append([x, y])
        elif f == 1:
            hull_right.append([x, y])
    hull_left = np.array(hull_left, np.int32)
    hull_right = np.array(hull_right, np.int32)
    return hull_left, hull_right


def get_sub_lines(in_points, gap_dis=3, min_line_len=5):
    sub_lines = []
    k, b = linear_regression(in_points, method=2)
    ll = params_to_line(k, b)
    l_points = []
    for p in in_points:
        p_ = project_point_to_line(p, ll)
        l_points.append(p_)
    # print(len(l_points))
    l_points = sorted(l_points, key=lambda cus: cus[1], reverse=False)
    s_point = l_points[0]
    e_point = l_points[0]
    for i in range(len(l_points) - 1):
        p1 = l_points[i]
        p2 = l_points[i + 1]
        dis = math.fabs(p1[1] - p2[1])
        if dis > gap_dis:
            if not s_point == e_point:
                if calc_y_len_on_line([s_point, e_point]) > min_line_len:
                    sub_lines.append([s_point, e_point])
            s_point = p2
            e_point = p2
        else:
            e_point = p2
    if not s_point == e_point:
        if calc_y_len_on_line([s_point, e_point]) > min_line_len:
            sub_lines.append([s_point, e_point])
    # print(sub_lines)
    return sub_lines


def calc_y_len_on_line(line):
    p1, p2 = line
    return math.fabs(p1[1] - p2[1])


def calc_y_len_on_sub_lines(sub_lines):
    d = 0.0
    for line in sub_lines:
        d += calc_y_len_on_line(line)
    return d


def get_line_occupancy(refer_line, sub_lines, h, w):
    sub_lines_y_len = calc_y_len_on_sub_lines(sub_lines)
    y_len_total = calc_y_len_on_line(refer_line)
    c_point = calc_cross_point(sub_lines[0], [[w, 0], [w, h]])
    y_len_total = min(y_len_total, calc_y_len_on_line([refer_line[0], c_point]))
    occupancy = sub_lines_y_len / y_len_total
    return occupancy


def get_line_occupancy_self(self_line, sub_lines, h, w):
    sub_lines_y_len = calc_y_len_on_sub_lines(sub_lines)
    y_len_total = calc_y_len_on_line(self_line)
    occupancy = sub_lines_y_len / y_len_total
    return occupancy


def remake_dotted_line(sub_lines):
    new_sub_lines = []
    temp_sub_lines = []
    for line in sub_lines:
        temp_sub_lines.append([line, line_dis(line)])
    temp_sub_lines = sorted(temp_sub_lines, key=lambda cus: cus[1], reverse=True)
    now_line, now_d = temp_sub_lines[0]
    new_sub_lines.append(now_line)
    for line, d in temp_sub_lines[1:]:
        now_p1, now_p2 = now_line
        p1, p2 = line
        if now_p1[1] < p2[1]:
            continue
        d2 = dis_between_two_points(p2, now_p1)
        if d2 < d or now_d <= d:
            continue
        now_d = d
        now_line = line
        new_sub_lines.append(now_line)
    return new_sub_lines


def remake_dotted_line_new(sub_lines,h,w):
    new_sub_lines = []
    temp_sub_lines = []
    border_gap_default = 10
    flag = True
    for line in sub_lines:
        p1, p2 = line
        x1, y1 = p1
        x2, y2 = p2
        if y1>h-border_gap_default or y2>h-border_gap_default or x1>w-border_gap_default or x2>w-border_gap_default:
            continue
        d = line_dis(line)
        if d < 20:
            continue
        temp_sub_lines.append([line, line[0][1], d])
    temp_sub_lines = sorted(temp_sub_lines, key=lambda cus: cus[1], reverse=True)
    now_line,_,now_d= temp_sub_lines[0]

    new_sub_lines.append(now_line)
    for line, _, d in temp_sub_lines[1:]:
        now_p1, now_p2 = now_line
        p1, p2 = line
        d2 = dis_between_two_points(p2, now_p1)
        if d2 < d or now_d <= d:
            break
        now_d = d
        now_line = line
        new_sub_lines.append(now_line)
    if len(new_sub_lines) < 3:
        flag = False
    return flag,new_sub_lines


def get_line_type_and_remake_sub_lines(sub_lines, h, w, refer_line):
    y_start, y_end = refer_line[0][1], refer_line[1][1]
    occupancy = get_line_occupancy(refer_line, sub_lines, h, w)
    self_line = [sub_lines[0][0],sub_lines[-1][1]]
    occupancy_self = get_line_occupancy_self(self_line, sub_lines, h, w)
    new_sub_line = []
    if occupancy < 0.2:
        line_type = -1
    else:
        line = sub_lines[0]
        k, b = line_to_params(line)
        x_start = (y_start - b) / k
        x_end = (y_end - b) / k
        new_sub_line = [[[x_start, y_start], [x_end, y_end]]]
        if occupancy > 0.5 or occupancy_self > 0.6:
            line_type = 0
        else:
            # tmp_sub_line = remake_dotted_line(sub_lines)
            flag, tmp_sub_line = remake_dotted_line_new(sub_lines,h,w)
            occupancy2 = get_line_occupancy(refer_line, tmp_sub_line, h, w)
            if occupancy2 < 0.2 or not flag:
                line_type = 0
            else:
                line_type = 1
                new_sub_line += tmp_sub_line
    return line_type, new_sub_line


def get_total_line_points(choose_points, pts, sigma=None, gradation_scope=None):
    candidate_points = []
    in_points = []
    for p in pts:
        dis = point_distance_line(p, choose_points)
        if math.fabs(dis) < calc_sigma(p, sigma, gradation_scope):
            in_points.append(p)
        else:
            candidate_points.append(p)
    return in_points, candidate_points


def sort_sub_lines_info(sub_lines_info):
    new_sub_lines_info = []
    tmp_sub_lines_info = []
    for data in sub_lines_info:
        line = data[0]
        p1, p2 = line
        tmp_sub_lines_info.append([data, p1[0]])
    tmp_sub_lines_info = sorted(tmp_sub_lines_info, key=lambda cus: cus[1], reverse=False)
    for data, _ in tmp_sub_lines_info:
        new_sub_lines_info.append(data)
    return new_sub_lines_info


def lane_line_extraction_v2(file_name,
                            hull_points,
                            direction=0,
                            visualization=False):
    # line_extraction_config = load_config(file_name)['line_extraction']
    # extra_boxes = load_extra_boxes(line_extraction_config['extra_boxes'])
    # ransac_sigma = line_extraction_config['ransac_sigma']
    # border_gap_default = line_extraction_config['border_gap']
    border_gap_default = 2
    path_dict = get_path_dict(file_name)
    img_path = path_dict['cal-img']
    img = cv2.imread(img_path)
    config = load_config(file_name)
    is_reverse = config['reverse']
    if is_reverse:
        img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    choose_hull = hull_points[direction]
    dst = get_mask_area_img(img, choose_hull)
    img_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    hull_lines = [[choose_hull[0], choose_hull[3]], [choose_hull[1], choose_hull[2]]]
    cross_point = calc_cross_point(hull_lines[0], hull_lines[1])
    h, w, _ = img.shape
    sub_lines_info = []

    edge = cv2.Canny(img_gray, 400, 400)
    # edge = cv2.Canny(img_gray, 300, 300)
    edge_pts = get_points_from_arr(edge)
    # edge_pts = get_points_from_arr(edge, extra_boxes=extra_boxes)
    edge_pts = check_area_points(edge_pts, choose_hull, border_gap_default=border_gap_default)


    edge_300 = cv2.Canny(img_gray, 300, 300)
    # edge_300 = cv2.Canny(img_gray, 200, 200)
    edge_pts_300 = get_points_from_arr(edge_300)
    # edge_pts_300 = get_points_from_arr(edge_300, extra_boxes=extra_boxes)
    edge_pts_300 = check_area_points(edge_pts_300, choose_hull, border_gap_default=border_gap_default)

    pts_div, img_list_ = divide_pts(edge_pts, choose_hull,visualization=visualization)
    candidate_points = edge_pts_300
    img_1 = img.copy()
    for idx,pts in enumerate(pts_div):
        if len(pts) == 0:
            continue
        sub_lines = get_sub_lines(pts)
        pts, candidate_points = get_total_line_points(sub_lines[0],
                                                      candidate_points,
                                                      sigma=[0.0, 20],
                                                      gradation_scope=[cross_point[1], h])
        sub_lines = get_sub_lines(pts)
        line_type, sub_lines = get_line_type_and_remake_sub_lines(sub_lines, h, w, hull_lines[0])
        if line_type != -1:
            sub_lines_info.append(sub_lines)
        if visualization:
            draw_pts_on_img(img_1, pts, point_weight=1, cc=get_color(idx))
            if line_type == 0:
                draw_lines_on_img(img,sub_lines)
            else:
                draw_lines_on_img(img,sub_lines[1:])
    sub_lines_info = sort_sub_lines_info(sub_lines_info)
    img_list = []
    if visualization:
        dst_ = dst.copy()
        draw_pts_on_img(dst_, edge_pts_300, point_weight=1, cc=get_color(1))
        draw_pts_on_img(dst_, edge_pts,point_weight=1)
        # if extra_boxes is not None:
        #     for box in extra_boxes:
        #         draw_box_on_img(dst_,box)
        img_list.append(dst_)
        img_list += img_list_
        img_list.append(img_1)
        img_list.append(img)
    return sub_lines_info, img_list


def lane_line_extraction(file_name, visualization=False):
    logger = get_logger('lane-line-extraction')
    logger.info('lane line extraction start...')
    t0 = time.time()
    path_dict = get_path_dict(file_name)
    lane_lines_path = path_dict[f'lane-lines']
    lane_lines_info = []
    hull_points = read_hull_points(file_name, logger)
    config = load_config(file_name)
    is_reverse = config['reverse']
    for d in [0, 1]:
        sub_lines_info, img_list = lane_line_extraction_v2(file_name,
                                                           hull_points,
                                                           direction=d,
                                                           visualization=visualization)
        if visualization:
            visualization_dir = path_dict['data-visualization-dir']
            for idx,img_ in enumerate(img_list):
                if is_reverse:
                    img_ = cv2.flip(img_, 1)
                lane_line_extraction_visualization_path = os.path.join(visualization_dir,
                                                                       f's4_lane_line_extraction_direction_{d}_{idx}.png')
                cv2.imwrite(lane_line_extraction_visualization_path, img_)
                logger.info(f'lane_line_extraction visualization ===> save to \'{lane_line_extraction_visualization_path}\'')
        lane_lines_info.append(sub_lines_info)
    t1 = time.time()
    logger.info(f'lane line extraction finished, use-time={round(t1 - t0, 3)}s')

    save_lane_lines(lane_lines_path, lane_lines_info, logger)


if __name__ == '__main__':
    args = make_args()
    lane_line_extraction(args.name,visualization=not args.non_visual)
