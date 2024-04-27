import torch


def det_img_pre_process(device,half,img):
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    return img
