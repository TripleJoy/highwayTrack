import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from track.track_tools.TrackerArgs import *
from track.track_tools.byteTracker import BYTETracker
from det.yolov5.utils.torch_utils import *
from tools.tools_init import *


@torch.no_grad()
def vehicle_pre_track(file_name):
    logger = get_logger('vehicle-pre-track')
    logger.info('vehicle pre track start...')
    t0 = time.time()
    path_dict = get_path_dict(file_name)
    track_labels_path = path_dict['pre-track-labels']
    config = load_config(file_name)
    track_config = config['track_config']
    img_size = config['img_size']
    # logger.info(f' track_config={track_config}')
    b_args = TrackerArgs(track_thresh=track_config['track_thresh'],
                         track_buffer=track_config['track_buffer'],
                         match_thresh=track_config['match_thresh'],
                         aspect_ratio_thresh=track_config['aspect_ratio_thresh'],
                         min_box_area=track_config['min_box_area'],
                         img_height=img_size[0],
                         img_width=img_size[0],
                         mot20=track_config['mot20'])
    byteTracker = BYTETracker(b_args, frame_rate=track_config['frame_rate'])
    det_info, max_frame = read_det_info(file_name,logger)
    track_info = {}
    for frame in tqdm(range(max_frame), ncols=100):
        if frame in det_info.keys():
            det = det_info[frame]
        else:
            det = []
        track_data = byteTracker.update(det)
        for t in track_data:
            x1, y1, x2, y2 = t.tlbr
            tid = t.track_id
            score = t.score
            cls = t.cls
            if tid in track_info.keys():
                track_info[tid].append(
                    [frame + 1, int(x1), int(y1), int(x2), int(y2), round(float(score), 3), int(cls)])
            else:
                track_info[tid] = [[frame + 1, int(x1), int(y1), int(x2), int(y2), round(float(score), 3), int(cls)]]
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
    logger.info(f'vehicle pre track finished, use-time={round(t1 - t0, 2)}s, vehicle-num={len(track_info.keys())}, '
                f'frames={max_frame}, time-per-frame={round((t1 - t0) / max_frame, 4)}s')
    save_pre_track_labels(track_labels_path, track_data, logger)


if __name__ == '__main__':
    args = make_args()
    vehicle_pre_track(args.name)
