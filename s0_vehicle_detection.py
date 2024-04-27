import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from det.yolov5.utils.datasets import LoadImages
from det.yolov5.utils.torch_utils import *
from det.det_tools.load_model import *
from det.det_tools.det_img_pre_process import *
from det.det_tools.det_and_nms import *
from det.det_tools.cfg_utils import *
from tools.tools_init import *
cv2.setNumThreads(-1)


@torch.no_grad()
def vehicle_detection(file_name):
    logger = get_logger('vehicle-detection')
    t0 = time.time()
    path_dict = get_path_dict(file_name)
    weights = path_dict['det-model']
    source = path_dict['video']
    det_labels_path = path_dict['det-labels']
    config = load_config(file_name)
    det_config = config['det_config']
    # logger.info(f'det_config={det_config}')
    imgsz = det_config['imgsz']
    device = det_config['device']
    conf_thres = det_config['conf_thres']
    iou_thres = det_config['iou_thres']
    max_det = det_config['max_det']
    classes = det_config['classes']
    cls_gap = 3
    for i in range(len(classes)):
        classes[i]+=cls_gap
    cls_min_box_area = det_config['cls_min_box_area']
    half = det_config['half']
    max_frame = det_config['max_frame']

    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model, names, imgsz, stride = load_model(weights, device, half, imgsz)
    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True, left_padding=0)
    videoCapture = cv2.VideoCapture(source)
    video_frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    max_frame = min(video_frames, max_frame)
    if device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once

    det_data = []
    t1= time.time()
    logger.info(f'loading det model and dataset finished, use-time={round(t1-t0,2)}s')
    logger.info('vehicle detection start...')
    with tqdm(total=max_frame,ncols=100) as bar:
        for path, img, im0s, r_img0, vid_cap, s in dataset:
            p, im0, r_im0, frame = Path(path), im0s.copy(), r_img0.copy(), getattr(dataset, 'frame', 0)
            if frame > max_frame:
                break
            img = det_img_pre_process(device, half, img)
            det, det_num = det_and_nms(model, img, conf_thres,
                                       iou_thres, classes, max_det,
                                       im0, cls_min_box_area,cls_gap)

            for det_info in det:
                x1, y1, x2, y2, conf, cls = det_info
                det_data.append([frame, int(x1), int(y1), int(x2), int(y2), round(float(conf),3), int(cls)])
            bar.update(1)
    t2 = time.time()
    logger.info(f'vehicle detection finished, use-time={round(t2 - t1, 2)}s, '
                f'frames={max_frame}, time-per-frame={round((t2 - t1) / max_frame, 2)}s')
    save_det_labels(det_labels_path, det_data, logger)


if __name__ == '__main__':
    args = make_args()
    vehicle_detection(args.name)
