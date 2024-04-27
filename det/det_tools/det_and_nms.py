import torch

from det.yolov5.utils.general import (non_max_suppression, scale_coords)


def det_and_nms(model, img, conf_thres,
                iou_thres, classes, max_det,
                im0,cls_min_box_area,cls_gap):
    pred = model(img)[0]

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, max_det=max_det)

    # check
    det = pred[0]
    det_num = len(det)
    det_check = []
    if det_num:
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        for *xyxy, conf, cls in reversed(det):
            x1 = int(xyxy[0])
            x2 = int(xyxy[2])
            y1 = int(xyxy[1])
            y2 = int(xyxy[3])
            cls -= cls_gap
            if (x2 - x1) * (y2 - y1) < cls_min_box_area[int(cls)]:
                continue
            det_check.append([x1, y1, x2, y2, float(conf), int(cls)])
        det_check = torch.Tensor(det_check)
    return det_check, det_num
