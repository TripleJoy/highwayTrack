import argparse
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0].parents[0]  # YOLOv5 root directory


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'det/yolov5/weights/yolov5l.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'source', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--file_name', default='exp', help='save results to project/name')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # print_args(FILE.stem, opt)
    return opt


# if __name__ == '__main__':
#     print(FILE)
#     print(ROOT)
