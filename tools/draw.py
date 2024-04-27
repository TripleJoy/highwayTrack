import cv2
import numpy as np


def get_color(idx):
    color_list = [
        (0, 165, 255),
        (255, 255, 51),
        (111,127,250),
        (90,169,89),
        (181,114,117),
        (70,23,11),
        (183, 165, 127),
        (0, 255, 0)
    ]
    return color_list[idx % len(color_list)]


def reg_pts(pts):
    new_pts = []
    for pt in pts:
        new_pts.append([int(round(pt[0])), int(round(pt[1]))])
    return new_pts


def reg_pts_float(pts,r=2):
    new_pts = []
    for pt in pts:
        new_pts.append([round(pt[0],r),round(pt[1],r)])
    return new_pts


def reg_pt(pt):
    return [int(round(pt[0])), int(round(pt[1]))]


def reg_pt_float(pt,r=2):
    return [round(pt[0],r),round(pt[1],r)]


def reg_line(l):
    return [reg_pt(l[0]), reg_pt(l[1])]


def draw_pts_on_img(img, pts, cc=(0, 165, 255),point_weight=5):
    for pt in pts:
        cv2.circle(img, reg_pt(pt), point_weight, cc, -1)


def draw_line_on_img(img, line, cc=(0, 165, 255), lc=(255, 255, 51),line_weight=3,endpoint_weight=5,show_endpoint=False):
    line = reg_line(line)
    cv2.line(img, line[0], line[1], lc, line_weight, cv2.LINE_AA)
    if show_endpoint:
        cv2.circle(img, line[0], endpoint_weight, cc, -1)
        cv2.circle(img, line[1], endpoint_weight, cc, -1)


def draw_lines_on_img(img, lines, cc=(0, 165, 255), lc=(255, 255, 51),line_weight=3,endpoint_weight=5,show_endpoint=False):

    for line in lines:
        draw_line_on_img(img, line, cc=cc, lc=lc,line_weight=line_weight,endpoint_weight=endpoint_weight,show_endpoint=show_endpoint)


def draw_box_on_img(img, box,lc=(255, 255, 51),line_weight=3):
    p1,p3 = [box[0],box[1]],[box[2],box[3]]
    p2 = [p1[0],p3[1]]
    p4 = [p3[0],p1[1]]
    draw_line_on_img(img,[p1,p2],lc=lc,line_weight=line_weight)
    draw_line_on_img(img,[p2,p3],lc=lc,line_weight=line_weight)
    draw_line_on_img(img,[p3,p4],lc=lc,line_weight=line_weight)
    draw_line_on_img(img,[p4,p1],lc=lc,line_weight=line_weight)


def draw_text_on_img(img, text, pos, size=0.75, color=(255, 255, 51), weight=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, weight)


def show_img(img,resize=None,is_wait=True):
    if resize is not None:
        img = cv2.resize(img, resize, interpolation=cv2.INTER_LINEAR)
    cv2.namedWindow("image")
    cv2.moveWindow('image', 0, 0)
    cv2.imshow("image", img)
    if is_wait:
        cv2.waitKey(0)
    else:
        cv2.waitKey(40)



def draw_area_on_img(im0, pts, color=(255, 255, 51), weight=2):
    pts = np.array(reg_pts(pts),dtype=np.int32)
    cv2.polylines(im0, [pts], True, color, weight)  # True表示该图形为封闭图形


def draw_box3d_on_img(im0, box3d, lc=(255, 255, 51), weight=2):
    box3d = np.array(reg_pts(box3d),dtype=np.int32)
    p1, p2, p3, p4, p5, p6, p7, p8 = box3d
    lines = [[p1,p2],[p2,p3],[p3,p4],[p4,p1],
             [p5,p6],[p6,p7],[p7,p8],[p8,p5],
             [p1,p5],[p2,p6],[p3,p7],[p4,p8],]
    draw_lines_on_img(im0,lines,line_weight=2,lc=lc)



