import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from pathlib import Path
from det.yolov5.models.experimental import attempt_load
import torch
from det.yolov5.utils.general import (check_suffix,check_img_size)


def load_model(weights,device,half,imgsz):
    w = str(weights[0] if isinstance(weights, list) else weights)
    suffix, suffixes = Path(w).suffix.lower(), ['.pt']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16
    imgsz = check_img_size(imgsz, s=stride)
    return model, names, imgsz, stride