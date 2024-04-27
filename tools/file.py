import os
import yaml
from tools.draw import *
import time


def get_path_dict(file_name_):
    file_name = file_name_.split('.')[0]
    path_dict = {
        'file_name': file_name,
        'source-dir': f'.\\source\\{file_name}',
        'middle-file-dir': f'.\\middle_file\\{file_name}',
        'eval-dir': f'.\\middle_file\\{file_name}\\eval',
        'data-visualization-dir': f'.\\data_visualization\\{file_name}',
        'models-dir': f'.\\models',

        'det-model': f'.\\models\\det\\yolov5l.pt',
        'seg-model': f'.\\models\\seg\\sam_vit_h_4b8939.pth',
        'seg-model-b': f'.\\models\\seg\\sam_vit_b_01ec64.pth',
        'seg-model-l': f'.\\models\\seg\\sam_vit_l_0b3195.pth',
        'base-config': '.\\config\\base_config.yaml',
        'config': f'.\\source\\{file_name}\\config.yaml',
        'gt-ori': f'.\\source\\{file_name}\\{file_name}_gt.txt',
        'gt-final': f'.\\middle_file\\{file_name}\\eval\\gt\\exp-test\\{file_name}\\gt.txt',
        'seq-info': f'.\\middle_file\\{file_name}\\eval\\gt\\exp-test\\{file_name}\\seqinfo.ini',
        'seq-maps': f'.\\middle_file\\{file_name}\\eval\\gt\\seqmaps\\exp-test.txt',
        'gt-folder': f'.\\middle_file\\{file_name}\\eval\\gt',
        'highwayTrack-online-folder': f'.\\middle_file\\{file_name}\\eval\\trackers\\highwayTrack\\online',
        'highwayTrack-base-interpolation-folder': f'.\\middle_file\\{file_name}\\eval\\trackers\\highwayTrack\\base-interpolation',
        'highwayTrack-online-res': f'.\\middle_file\\{file_name}\\eval\\res\\highwayTrack\\online',
        'highwayTrack-base-interpolation-res': f'.\\middle_file\\{file_name}\\eval\\res\\highwayTrack\\base-interpolation',
        'byteTrack-online-folder': f'.\\middle_file\\{file_name}\\eval\\trackers\\byteTrack\\online',
        'byteTrack-base-interpolation-folder': f'.\\middle_file\\{file_name}\\eval\\trackers\\byteTrack\\base-interpolation',
        'byteTrack-online-res': f'.\\middle_file\\{file_name}\\eval\\res\\byteTrack\\online',
        'byteTrack-base-interpolation-res': f'.\\middle_file\\{file_name}\\eval\\res\\byteTrack\\base-interpolation',

        'video': f'.\\source\\{file_name}\\{file_name_}',
        'cal-img': f'.\\middle_file\\{file_name}\\cal-img.png',
        'det-labels': f'.\\middle_file\\{file_name}\\s0-det-labels.txt',
        'pre-track-labels': f'.\\middle_file\\{file_name}\\s1-pre-track-labels.txt',
        'lane-area-mark-points': f'.\\middle_file\\{file_name}\\s2-lane-area-mark-points.txt',
        'hull-points': f'.\\middle_file\\{file_name}\\s3-hull-points.txt',
        'lane-lines': f'.\\middle_file\\{file_name}\\s4-lane-lines.txt',
        'cal-points': f'.\\middle_file\\{file_name}\\s4-cal-points.txt',
        'cal-params': f'.\\middle_file\\{file_name}\\s5-cal-params.txt',
        'lane-lines-completion': f'.\\middle_file\\{file_name}\\s6-lane-lines-completion.txt',
        # 'vehicle-boundary-points': f'.\\middle_file\\{file_name}\\vehicle-boundary-points.txt',
        # 'vehicle-hull-points': f'.\\middle_file\\{file_name}\\08-vehicle-hull-points.txt',
        # 'vehicle-seg-infos-init': f'.\\middle_file\\{file_name}\\09-vehicle-seg-infos-init.txt',
        # 'vehicle-seg-infos-init-1': f'.\\middle_file\\{file_name}\\09-vehicle-seg-infos-init-1.txt',
        'highwayTrack-online': f'.\\middle_file\\{file_name}\\s7-highwayTrack-online.txt',
        'highwayTrack-base-interpolation': f'.\\middle_file\\{file_name}\\s7-highwayTrack-base-interpolation.txt',
        # 'byteTrack-realtime': f'.\\middle_file\\{file_name}\\11-byteTrack-realtime.txt',
        # 'byteTrack-base-interpolation': f'.\\middle_file\\{file_name}\\11-byteTrack-base-interpolation.txt',
        'highwayTrack-online-mot17': f'.\\middle_file\\{file_name}\\eval\\trackers\\highwayTrack\\online\\exp-test\\{file_name}\\{file_name}.txt',
        'highwayTrack-base-interpolation-mot17': f'.\\middle_file\\{file_name}\\eval\\trackers\\highwayTrack\\base-interpolation\\exp-test\\{file_name}\\{file_name}.txt',
        # 'byteTrack-realtime-mot17': f'.\\middle_file\\{file_name}\\eval\\trackers\\byteTrack\\realtime\\exp-test\\{file_name}\\{file_name}.txt',
        # 'byteTrack-base-interpolation-mot17': f'.\\middle_file\\{file_name}\\eval\\trackers\\byteTrack\\base-interpolation\\exp-test\\{file_name}\\{file_name}.txt',
    }
    os.makedirs(path_dict['source-dir'], exist_ok=True)
    os.makedirs(path_dict['middle-file-dir'], exist_ok=True)
    os.makedirs(path_dict['data-visualization-dir'], exist_ok=True)
    return path_dict


def load_config(file_name):
    path_dict = get_path_dict(file_name)
    video_path = path_dict[f'video']
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with open(path_dict['base-config'], 'r', encoding='utf-8') as f:
        data = f.read()
        config = yaml.load(data, Loader=yaml.FullLoader)
    if os.path.exists(path_dict['config']):
        with open(path_dict['config'], 'r', encoding='utf-8') as f:
            data = f.read()
            config_tmp = yaml.load(data, Loader=yaml.FullLoader)
        for key in config_tmp.keys():
            config[key] = config_tmp[key]
    config['total_frames'] = total_frames
    return config


def load_extra_boxes(extra_boxes_data=None):
    if extra_boxes_data is None:
        return None
    num = len(extra_boxes_data) // 4
    extra_boxes = []
    for i in range(num):
        x = i * 4
        extra_boxes.append(
            [extra_boxes_data[x], extra_boxes_data[x + 1], extra_boxes_data[x + 2], extra_boxes_data[x + 3]])
    return extra_boxes


def get_cal_img(file_name, logger):
    path_dict = get_path_dict(file_name)
    cal_img_path = path_dict[f'cal-img']
    logger.info(f'get cal-img from \'{cal_img_path}\'')
    img = cv2.imread(cal_img_path)
    return img


def get_video(file_name, logger):
    path_dict = get_path_dict(file_name)
    video_path = path_dict[f'video']
    logger.info(f'get video from \'{video_path}\'')
    cap = cv2.VideoCapture(video_path)
    return cap


def read_list_from_txt(file_path):
    f = open(file_path, 'r')
    line = f.readline()  # 调用文件的 readline()方法
    data = []
    while line:
        if line[0] == '#':
            line = f.readline()
            continue
        # print(line, end='')
        d = line.split('\n')[0].split(' ')
        data.append([float(x) for x in d])
        line = f.readline()
    f.close()
    return data


def check_txt_exist(file_path):
    return os.path.exists(file_path)


def check_box_data(box, img_size, border_gap):
    x1, y1, x2, y2 = box
    w, h = img_size
    if border_gap < x1 < w - border_gap and border_gap < x2 < w - border_gap \
            and border_gap < y1 < h - border_gap and border_gap < y2 < h - border_gap:
        return True
    else:
        return False


def read_det_info(file_name, logger, cls_choose=None):
    path_dict = get_path_dict(file_name)
    det_labels_path = path_dict[f'det-labels']
    logger.info(f'read det-labels from \'{det_labels_path}\'')
    if cls_choose is None:
        cls_choose = [0, 1, 2]
    det_labels = read_list_from_txt(det_labels_path)
    det_info = {}
    max_frame = 0
    for data in det_labels:
        frame, x1, y1, x2, y2, conf, cls = data
        frame = int(frame) - 1
        cls = int(cls)
        max_frame = max(max_frame, frame)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if cls not in cls_choose:
            continue
        if frame in det_info.keys():
            det_info[frame].append([x1, y1, x2, y2, conf, cls])
        else:
            det_info[frame] = [[x1, y1, x2, y2, conf, cls]]

    return det_info, max_frame


def read_det_info_in_lane_area(file_name, logger):
    path_dict = get_path_dict(file_name)
    hull_points = read_hull_points(file_name, logger)
    det_labels_path = path_dict[f'det-labels']
    logger.info(f'read det-labels-in-lane-area from \'{det_labels_path}\'')
    y_limit = hull_points[0][0][1]
    config = load_config(file_name)
    track_config = config['track_config']
    cls_choose = track_config['cls_choose']
    if cls_choose is None:
        cls_choose = [0, 1, 2]
    det_labels = read_list_from_txt(det_labels_path)
    det_info = {}
    max_frame = 0
    for data in det_labels:
        frame, x1, y1, x2, y2, conf, cls = data
        frame = int(frame) - 1
        cls = int(cls)
        max_frame = max(max_frame, frame)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if not check_box_data([x1, y1, x2, y2], config['img_size'], config['border_gap']):
            continue
        if cls not in cls_choose or y2 < y_limit:
            continue
        if frame in det_info.keys():
            det_info[frame].append([x1, y1, x2, y2, conf, cls])
        else:
            det_info[frame] = [[x1, y1, x2, y2, conf, cls]]

    return det_info, max_frame


def read_pre_track_labels(file_name, logger):
    path_dict = get_path_dict(file_name)
    pre_track_labels_path = path_dict[f'pre-track-labels']
    config = load_config(file_name)
    is_reverse = config['reverse']
    track_config = config['track_config']
    img_size = config['img_size']
    cls_choose = track_config['cls_choose']
    if cls_choose is None:
        cls_choose = [0, 1, 2]
    logger.info(f'read pre-track-labels from \'{pre_track_labels_path}\'')
    pre_track_data = read_list_from_txt(pre_track_labels_path)
    pre_track_labels_by_fid = {}
    pre_track_labels_by_vid = {}
    max_frame = 0
    for data in pre_track_data:
        vid, fid, x1, y1, x2, y2, conf, cls = data
        if not check_box_data([x1, y1, x2, y2], img_size, config['border_gap']):
            continue
        if is_reverse:
            x1 = img_size[0] - 1 - x1
            x2 = img_size[0] - 1 - x2
            x1, x2 = x2, x1
        vid = int(vid)
        fid = int(fid) - 1
        cls = int(cls)
        max_frame = max(max_frame, fid)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if cls not in cls_choose:
            continue
        if fid in pre_track_labels_by_fid.keys():
            pre_track_labels_by_fid[fid].append([vid, x1, y1, x2, y2, conf, cls])
        else:
            pre_track_labels_by_fid[fid] = [[vid, x1, y1, x2, y2, conf, cls]]
        if vid in pre_track_labels_by_vid.keys():
            pre_track_labels_by_vid[vid].append([fid, x1, y1, x2, y2, conf, cls])
        else:
            pre_track_labels_by_vid[vid] = [[fid, x1, y1, x2, y2, conf, cls]]

    return pre_track_labels_by_fid, pre_track_labels_by_vid, max_frame


def read_highway_track_online(file_name, logger):
    path_dict = get_path_dict(file_name)
    highway_track_labels_path = path_dict[f'highwayTrack-online']
    config = load_config(file_name)
    is_reverse = config['reverse']
    track_config = config['track_config']
    img_size = config['img_size']
    cls_choose = track_config['cls_choose']
    if cls_choose is None:
        cls_choose = [0, 1, 2]
    logger.info(f'read highway track online from \'{highway_track_labels_path}\'')
    track_data = read_list_from_txt(highway_track_labels_path)
    track_labels_by_fid = {}
    track_labels_by_vid = {}
    max_frame = 0
    for data in track_data:
        vid, fid, x1, y1, x2, y2, conf, cls, speed, is_predict = data
        if not check_box_data([x1, y1, x2, y2], img_size, config['border_gap']):
            continue
        # if is_reverse:
        #     x1 = img_size[0] - 1 - x1
        #     x2 = img_size[0] - 1 - x2
        #     x1, x2 = x2, x1
        vid = int(vid)
        fid = int(fid)
        cls = int(cls)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        is_predict = int(is_predict)
        max_frame = max(max_frame, fid)
        if cls not in cls_choose:
            continue
        if fid in track_labels_by_fid.keys():
            track_labels_by_fid[fid].append([vid, x1, y1, x2, y2, conf, cls, speed, is_predict])
        else:
            track_labels_by_fid[fid] = [[vid, x1, y1, x2, y2, conf, cls, speed, is_predict]]
        if vid in track_labels_by_vid.keys():
            track_labels_by_vid[vid].append([fid, x1, y1, x2, y2, conf, cls, speed, is_predict])
        else:
            track_labels_by_vid[vid] = [[fid, x1, y1, x2, y2, conf, cls, speed, is_predict]]

    return track_labels_by_fid, track_labels_by_vid, max_frame


def read_byte_track_online(file_name, logger):
    path_dict = get_path_dict(file_name)
    highway_track_labels_path = path_dict[f'byteTrack-online']
    config = load_config(file_name)
    is_reverse = config['reverse']
    track_config = config['track_config']
    img_size = config['img_size']
    cls_choose = track_config['cls_choose']
    if cls_choose is None:
        cls_choose = [0, 1, 2]
    logger.info(f'read byte track online from \'{highway_track_labels_path}\'')
    track_data = read_list_from_txt(highway_track_labels_path)
    track_labels_by_fid = {}
    track_labels_by_vid = {}
    max_frame = 0
    for data in track_data:
        vid, fid, x1, y1, x2, y2, conf, cls = data
        if not check_box_data([x1, y1, x2, y2], img_size, config['border_gap']):
            continue
        if is_reverse:
            x1 = img_size[0] - 1 - x1
            x2 = img_size[0] - 1 - x2
            x1, x2 = x2, x1
        vid = int(vid)
        fid = int(fid)
        cls = int(cls)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        max_frame = max(max_frame, fid)
        if cls not in cls_choose:
            continue
        if fid in track_labels_by_fid.keys():
            track_labels_by_fid[fid].append([vid, x1, y1, x2, y2, conf, cls])
        else:
            track_labels_by_fid[fid] = [[vid, x1, y1, x2, y2, conf, cls]]
        if vid in track_labels_by_vid.keys():
            track_labels_by_vid[vid].append([fid, x1, y1, x2, y2, conf, cls])
        else:
            track_labels_by_vid[vid] = [[fid, x1, y1, x2, y2, conf, cls]]

    return track_labels_by_fid, track_labels_by_vid, max_frame


def read_track_info_in_lane_area(file_name, logger):
    path_dict = get_path_dict(file_name)
    track_labels_path = path_dict[f'track-labels']
    hull_points = read_hull_points(file_name, logger)
    y_limit = hull_points[0][0][1]
    config = load_config(file_name)
    track_config = config['track_config']
    cls_choose = track_config['cls_choose']
    if cls_choose is None:
        cls_choose = [0, 1, 2]
    logger.info(f'read track-labels from \'{track_labels_path}\'')
    track_data = read_list_from_txt(track_labels_path)
    track_info_by_frame = {}
    track_info_by_vid = {}
    max_frame = 0
    for data in track_data:
        idx, frame, x1, y1, x2, y2, conf, cls = data
        if not check_box_data([x1, y1, x2, y2], config['img_size'], config['border_gap']):
            continue

        idx = int(idx)
        frame = int(frame) - 1
        cls = int(cls)

        max_frame = max(max_frame, frame)
        if cls not in cls_choose or y2 < y_limit:
            continue
        if frame in track_info_by_frame.keys():
            track_info_by_frame[frame].append([idx, x1, y1, x2, y2, conf, cls])
        else:
            track_info_by_frame[frame] = [[idx, x1, y1, x2, y2, conf, cls]]
        if idx in track_info_by_vid.keys():
            track_info_by_vid[idx].append([frame, x1, y1, x2, y2, conf, cls])
        else:
            track_info_by_vid[idx] = [[frame, x1, y1, x2, y2, conf, cls]]

    return track_info_by_frame, track_info_by_vid, max_frame


def read_lane_area_mark_points(file_name, logger):
    path_dict = get_path_dict(file_name)
    lane_area_mark_points_path = path_dict[f'lane-area-mark-points']
    logger.info(f'read lane-area-mark-points from \'{lane_area_mark_points_path}\'')
    lane_area_mark_points = read_list_from_txt(lane_area_mark_points_path)
    input_point, input_label = [], []
    for f, x, y in lane_area_mark_points:
        f, x, y = int(f), int(x), int(y)
        input_point.append([x, y])
        input_label.append(f)
    return [input_point, input_label]


def read_hull_points(file_name, logger):
    path_dict = get_path_dict(file_name)
    hull_points_path = path_dict['hull-points']
    logger.info(f'read hull-points from \'{hull_points_path}\'')
    hull_points = read_list_from_txt(hull_points_path)
    hull_points_remake = [[], []]
    for f, x, y in hull_points:
        f = int(f)
        hull_points_remake[f].append([x, y])
    hull_points_remake[0] = np.array(hull_points_remake[0], np.int32)
    hull_points_remake[1] = np.array(hull_points_remake[1], np.int32)
    return hull_points_remake


def read_lane_lines_info(file_name, logger):
    path_dict = get_path_dict(file_name)
    lane_lines_path = path_dict[f'lane-lines']
    logger.info(f'read lane-lines from \'{lane_lines_path}\'')
    lane_lines = [[], []]

    lane_lines_info = read_list_from_txt(lane_lines_path)

    for d, index, x1, y1, x2, y2 in lane_lines_info:
        d = int(d)
        index = int(index)
        if index == len(lane_lines[d]):
            lane_lines[d].append([])
        lane_lines[d][index].append([[x1, y1], [x2, y2]])

    return lane_lines


def read_lane_lines_completion_info(file_name, logger):
    path_dict = get_path_dict(file_name)
    lane_lines_completion_path = path_dict[f'lane-lines-completion']
    logger.info(f'read lane-lines-completion from \'{lane_lines_completion_path}\'')
    lane_lines_completion = []

    lane_lines_completion_info = read_list_from_txt(lane_lines_completion_path)

    for index, x1, y1, x2, y2 in lane_lines_completion_info:
        index = int(index)
        if index == len(lane_lines_completion):
            lane_lines_completion.append([])
        lane_lines_completion[index].append([[x1, y1], [x2, y2]])
    return lane_lines_completion


def read_cal_points(file_name, logger):
    path_dict = get_path_dict(file_name)
    cal_points_path = path_dict[f'cal-points']
    logger.info(f'read cal-points from \'{cal_points_path}\'')
    cal_points = [[], []]

    cal_points_data = read_list_from_txt(cal_points_path)

    for d, x1, y1, x2, y2 in cal_points_data:
        d = int(d)
        cal_points[d].append([[x1, y1], [x2, y2]])

    return cal_points


def read_cal_params(file_name, logger):
    path_dict = get_path_dict(file_name)
    cal_params_path = path_dict[f'cal-params']
    logger.info(f'read cal-params from \'{cal_params_path}\'')

    config = load_config(file_name)
    img_size = config['img_size']
    center_point = [img_size[0] // 2, img_size[1] // 2]
    ori_pt, vp1, f, h = read_list_from_txt(cal_params_path)
    ori_pt = [ori_pt[0] - center_point[0], ori_pt[1] - center_point[1]]
    vp1 = [vp1[0] - center_point[0], vp1[1] - center_point[1]]
    cal_params = [f[0], h[0], vp1]
    return ori_pt, cal_params


def read_vehicle_boundary_points(file_name, logger):
    path_dict = get_path_dict(file_name)
    vehicle_boundary_points_path = path_dict[f'vehicle-boundary-points']
    logger.info(f'read vehicle-boundary-points from \'{vehicle_boundary_points_path}\'')
    vehicle_boundary_points_data = read_list_from_txt(vehicle_boundary_points_path)
    v_b_pts_info = {}
    for fid, vid, x, y in vehicle_boundary_points_data:
        fid = int(fid)
        vid = int(vid)
        k = f'{fid}_{vid}'
        if k not in v_b_pts_info.keys():
            v_b_pts_info[k] = []
        v_b_pts_info[k].append([x, y])

    return v_b_pts_info


def read_vehicle_hull_points(file_name, logger):
    path_dict = get_path_dict(file_name)
    vehicle_hull_points_path = path_dict[f'vehicle-hull-points']
    logger.info(f'read vehicle-hull-points from \'{vehicle_hull_points_path}\'')
    vehicle_hull_points_data = read_list_from_txt(vehicle_hull_points_path)
    v_h_pts_info = {}
    for fid, bid, x, y in vehicle_hull_points_data:
        fid = int(fid)
        bid = int(bid)
        k = f'{fid}_{bid}'
        if k not in v_h_pts_info.keys():
            v_h_pts_info[k] = []
        v_h_pts_info[k].append([x, y])
    return v_h_pts_info


def read_vehicle_infos(file_name, logger):
    path_dict = get_path_dict(file_name)
    vehicle_infos_path = path_dict[f'vehicle-infos']
    logger.info(f'read vehicle-infos from \'{vehicle_infos_path}\'')
    vehicle_infos_data = read_list_from_txt(vehicle_infos_path)
    vehicle_infos_by_vid = {}
    for vid, fid, x1, y1, x2, y2, x1_, y1_, x2_, y2_, kp1_x, kp1_y, kp2_x, kp2_y, kp3_x, kp3_y, vehicle_length, vehicle_width, vehicle_pos_x, vehicle_pos_y in vehicle_infos_data:
        fid = int(fid)
        vid = int(vid)
        box_ori = [x1, y1, x2, y2]
        box_fix = [x1_, y1_, x2_, y2_]
        kp1 = [kp1_x, kp1_y]
        kp2 = [kp2_x, kp2_y]
        kp3 = [kp3_x, kp3_y]
        vehicle_pos = [vehicle_pos_x, vehicle_pos_y]
        if vid not in vehicle_infos_by_vid.keys():
            vehicle_infos_by_vid[vid] = []
        vehicle_infos_by_vid[vid].append(
            [fid, box_ori, box_fix, [kp1, kp2, kp3], vehicle_length, vehicle_width, vehicle_pos])
    return vehicle_infos_by_vid


def read_vehicle_seg_infos_init(file_name, logger):
    path_dict = get_path_dict(file_name)
    vehicle_seg_infos_init_path = path_dict[f'vehicle-seg-infos-init']
    logger.info(f'read vehicle-seg-infos-init from \'{vehicle_seg_infos_init_path}\'')
    vehicle_seg_infos_init_data = read_list_from_txt(vehicle_seg_infos_init_path)
    vehicle_seg_infos_init_by_fid = {}
    for fid, bid, x1, y1, x2, y2, x1_, y1_, x2_, y2_, kp1_x, kp1_y, kp2_x, kp2_y, kp3_x, kp3_y, \
        vehicle_length, vehicle_width, vehicle_pos_x, vehicle_pos_y, score, cls, flag in vehicle_seg_infos_init_data:
        fid = int(fid)
        bid = int(bid)
        box_ori = [x1, y1, x2, y2]
        box_real = [x1_, y1_, x2_, y2_]
        kp1 = [kp1_x, kp1_y]
        kp2 = [kp2_x, kp2_y]
        kp3 = [kp3_x, kp3_y]
        vehicle_pos = [vehicle_pos_x, vehicle_pos_y]
        cls = int(cls)
        flag = int(flag)
        if fid not in vehicle_seg_infos_init_by_fid.keys():
            vehicle_seg_infos_init_by_fid[fid] = []
        vehicle_seg_infos_init_by_fid[fid].append(
            [bid, box_ori, box_real, [kp1, kp2, kp3], vehicle_length, vehicle_width, vehicle_pos, score, cls, flag])
    return vehicle_seg_infos_init_by_fid


def clear_txt(file_path):
    open(file_path, mode='w').close()


def write_line_to_txt(file_path, data, mode='a'):
    f = open(file_path, mode)
    if data[0] != '#':
        line_data = ''
        for x in data:
            line_data += str(x) + ' '
        line_data = line_data.rstrip()
    else:
        line_data = data
    f.write(line_data + '\n')
    f.close()


def write_list_to_txt(file_path, data, mode='w'):
    f = open(file_path, mode)
    for line in data:
        line_data = ''
        for x in line:
            line_data += str(x) + ' '
        f.write(line_data.rstrip() + '\n')
    f.close()


def save_det_labels(det_labels_path, det_data, logger):
    t0 = time.time()
    clear_txt(det_labels_path)
    annotation = '# format: (frame_id, pt0_x, pt0_y, pt1_x, pt1_y, score, cls)'
    write_line_to_txt(det_labels_path, annotation)
    write_list_to_txt(det_labels_path, det_data)
    t1 = time.time()
    logger.info(f'===> save det-labels to \'{det_labels_path}\', use-time: {round(t1 - t0, 2)}s')


def save_pre_track_labels(pre_track_labels_path, track_labels, logger):
    t0 = time.time()
    clear_txt(pre_track_labels_path)
    annotation = '# format: (vehicle_id, frame, pt0_x, pt0_y, pt1_x, pt1_y, score, cls)'
    write_line_to_txt(pre_track_labels_path, annotation)
    write_list_to_txt(pre_track_labels_path, track_labels, mode='a')
    t1 = time.time()
    logger.info(f'===> save pre-track-labels to \'{pre_track_labels_path}\', use-time={round(t1 - t0, 2)}s')


def save_byte_track_labels(byte_track_labels_path, track_labels, logger):
    t0 = time.time()
    clear_txt(byte_track_labels_path)
    annotation = '# format: (vehicle_id, frame, pt0_x, pt0_y, pt1_x, pt1_y, score, cls)'
    write_line_to_txt(byte_track_labels_path, annotation)
    write_list_to_txt(byte_track_labels_path, track_labels, mode='a')
    t1 = time.time()
    logger.info(f'===> save byte_track_labelsto \'{byte_track_labels_path}\', use-time={round(t1 - t0, 2)}s')


def save_highway_track_real_labels(highway_track_real_labels_path, track_labels, logger):
    t0 = time.time()
    clear_txt(highway_track_real_labels_path)
    annotation = '# format: (vehicle_id, frame, pt0_x, pt0_y, pt1_x, pt1_y, score, cls, speed, is_predict)'
    write_line_to_txt(highway_track_real_labels_path, annotation)
    write_list_to_txt(highway_track_real_labels_path, track_labels, mode='a')
    t1 = time.time()
    logger.info(f'===> save highway-track-real-labels to \'{highway_track_real_labels_path}\', use-time={round(t1 - t0, 2)}s')


def save_highway_track_base_interpolation_labels(save_path, track_labels, logger):
    t0 = time.time()
    clear_txt(save_path)
    annotation = '# format: (vehicle_id, frame, pt0_x, pt0_y, pt1_x, pt1_y)'
    write_line_to_txt(save_path, annotation)
    write_list_to_txt(save_path, track_labels, mode='a')
    t1 = time.time()
    logger.info(f'===> save highway-track-base-interpolation-labels to \'{save_path}\', use-time={round(t1 - t0, 2)}s')


def save_byte_track_base_interpolation_labels(save_path, track_labels, logger):
    t0 = time.time()
    clear_txt(save_path)
    annotation = '# format: (vehicle_id, frame, pt0_x, pt0_y, pt1_x, pt1_y)'
    write_line_to_txt(save_path, annotation)
    write_list_to_txt(save_path, track_labels, mode='a')
    t1 = time.time()
    logger.info(f'===> save byte-track-base-interpolation-labels to \'{save_path}\', use-time={round(t1 - t0, 2)}s')


def save_lane_area_mark_points(lane_area_mark_points_path, lane_area_mark_points, logger):
    t0 = time.time()
    clear_txt(lane_area_mark_points_path)
    annotation = '# format: (direction, pt_x, pt_y)'
    write_line_to_txt(lane_area_mark_points_path, annotation)
    data = []
    for d in [0, 1]:
        pts = lane_area_mark_points[d]
        for pt in pts:
            data.append([d, int(pt[0]), int(pt[1])])
    write_list_to_txt(lane_area_mark_points_path, data, mode='a')
    t1 = time.time()
    logger.info(f'===> save lane-area-mark-points to \'{lane_area_mark_points_path}\', use-time={round(t1 - t0, 3)}s')


def save_hull_points(hull_points_path, hull_points, logger):
    t0 = time.time()
    clear_txt(hull_points_path)
    annotation = '# format: (direction, pt_x, pt_y)'
    write_line_to_txt(hull_points_path, annotation)
    data = []
    for d in [0, 1]:
        for pt in hull_points[d]:
            data.append([d, pt[0], pt[1]])
    write_list_to_txt(hull_points_path, data, mode='a')
    t1 = time.time()
    logger.info(f'===> save hull-points to \'{hull_points_path}\', use-time={round(t1 - t0, 4)}s')


def save_lane_lines(lane_lines_path, lane_lines_info, logger):
    t0 = time.time()
    clear_txt(lane_lines_path)
    annotation = '# format: (direction, line_idx, pt0_x, pt0_y, pt1_x, pt1_y)'
    write_line_to_txt(lane_lines_path, annotation)
    data = []
    for d in [0, 1]:
        sub_lines_info = lane_lines_info[d]
        index = 0
        for sub_lines in sub_lines_info:
            for line in sub_lines:
                p1, p2 = line
                x1, y1 = p1
                x2, y2 = p2
                data.append([d, index, round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)])
            index += 1
    write_list_to_txt(lane_lines_path, data, mode='a')
    t1 = time.time()
    logger.info(f'===> save lane-lines-info to \'{lane_lines_path}\', use-time={round(t1 - t0, 3)}s')


def save_lane_lines_completion(lane_lines_completion_path, lane_lines_completion, logger):
    t0 = time.time()
    clear_txt(lane_lines_completion_path)
    annotation = '# format: (line_idx, pt0_x, pt0_y, pt1_x, pt1_y)'
    write_line_to_txt(lane_lines_completion_path, annotation)
    data = []
    index = 0
    for lane_lines in lane_lines_completion:
        for sub_lines in lane_lines:
            p1, p2 = sub_lines
            x1, y1 = p1
            x2, y2 = p2
            data.append([index, round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)])
        index += 1
    write_list_to_txt(lane_lines_completion_path, data, mode='a')
    t1 = time.time()
    logger.info(
        f'===> save lane-lines-completion-info to \'{lane_lines_completion_path}\', use-time={round(t1 - t0, 3)}s')


def save_cal_points(cal_points_path, cal_points, logger):
    t0 = time.time()
    clear_txt(cal_points_path)
    annotation = '# format: (direction, img_x, img_y, world_x, world_y)'
    write_line_to_txt(cal_points_path, annotation)
    cal_points_data = []
    for d in [0, 1]:
        for cal in cal_points[d]:
            (x1, y1), (x2, y2) = cal
            cal_points_data.append([d, x1, y1, x2, y2])
    write_list_to_txt(cal_points_path, cal_points_data, mode='a')
    t1 = time.time()
    logger.info(f'===> save cal-points to \'{cal_points_path}\', use-time={round(t1 - t0, 3)}s')


def save_cal_params(cal_params_path, cal_params, logger):
    t0 = time.time()
    clear_txt(cal_params_path)
    annotation = '# params: ori_point_x, ori_point_y, vp1_x,vp1_y, f, h'
    write_line_to_txt(cal_params_path, annotation)
    ori_point, vp1, f, h = cal_params
    write_line_to_txt(cal_params_path, ori_point, mode='a')
    write_line_to_txt(cal_params_path, vp1, mode='a')
    write_line_to_txt(cal_params_path, [f], mode='a')
    write_line_to_txt(cal_params_path, [h], mode='a')
    t1 = time.time()
    logger.info(f'===> save cal-params to \'{cal_params_path}\', use-time={round(t1 - t0, 4)}s')


def save_vehicle_boundary_points(vehicle_boundary_points_path, vehicle_boundary_points, logger):
    t0 = time.time()
    clear_txt(vehicle_boundary_points_path)
    annotation = '# format: (frame_id, vehicle_id, pt_x, pt_y)'
    write_line_to_txt(vehicle_boundary_points_path, annotation)
    write_list_to_txt(vehicle_boundary_points_path, vehicle_boundary_points, mode='a')
    t1 = time.time()
    logger.info(
        f'===> save vehicle-boundary-points to \'{vehicle_boundary_points_path}\', use-time={round(t1 - t0, 3)}s')


def save_vehicle_hull_points(vehicle_hull_points_path, vehicle_hull_points, logger):
    t0 = time.time()
    clear_txt(vehicle_hull_points_path)
    annotation = '# format: (frame_id, box_id, pt_x, pt_y)'
    write_line_to_txt(vehicle_hull_points_path, annotation)
    write_list_to_txt(vehicle_hull_points_path, vehicle_hull_points, mode='a')
    t1 = time.time()
    logger.info(f'===> save vehicle-hull-points to \'{vehicle_hull_points_path}\', use-time: {round(t1 - t0, 3)}s')


def save_vehicle_infos(vehicle_infos_path, vehicle_infos, logger):
    t0 = time.time()
    clear_txt(vehicle_infos_path)
    annotation = '# format: (frame_id, vehicle_id,\n' \
                 '#          x1, y1, x2, y2,\n' \
                 '#          x1_, y1_, x2_, y2_,\n' \
                 '#          key_points_1_x, key_points_1_y,\n' \
                 '#          key_points_2_x, key_points_2_y,\n' \
                 '#          key_points_3_x, key_points_3_y,\n' \
                 '#          vehicle_length, vehicle_width,\n' \
                 '#          vehicle_pos_x, vehicle_pos_y)'
    write_line_to_txt(vehicle_infos_path, annotation)
    write_list_to_txt(vehicle_infos_path, vehicle_infos, mode='a')
    t1 = time.time()
    logger.info(f'===> save vehicle-infos to \'{vehicle_infos_path}\', use-time: {round(t1 - t0, 2)}s')


def save_vehicle_seg_infos_init(vehicle_seg_infos_init_path, vehicle_seg_infos_init, logger):
    t0 = time.time()
    clear_txt(vehicle_seg_infos_init_path)
    annotation = '# format: (frame_id, box_id,\n' \
                 '#          x1, y1, x2, y2,\n' \
                 '#          x1_, y1_, x2_, y2_,\n' \
                 '#          key_points_1_x, key_points_1_y,\n' \
                 '#          key_points_2_x, key_points_2_y,\n' \
                 '#          key_points_3_x, key_points_3_y,\n' \
                 '#          vehicle_length, vehicle_width,\n' \
                 '#          vehicle_pos_x, vehicle_pos_y,\n' \
                 '#          cls, score, flag)'
    write_line_to_txt(vehicle_seg_infos_init_path, annotation)
    write_list_to_txt(vehicle_seg_infos_init_path, vehicle_seg_infos_init, mode='a')
    t1 = time.time()
    logger.info(
        f'===> save vehicle-seg-infos-init to \'{vehicle_seg_infos_init_path}\', use-time: {round(t1 - t0, 2)}s')
