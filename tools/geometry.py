import math
import numpy as np
import cv2


def find_max_region(mask_sel):
    contours, hierarchy = cv2.findContours(mask_sel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # 找到最大区域并填充
    area = []
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
    max_idx = np.argmax(area)
    for k in range(len(contours)):

        if k != max_idx:
            cv2.fillPoly(mask_sel, [contours[k]], 0)
    boundary_points = []
    for pt in contours[max_idx]:
        boundary_points.append(pt[0])
    hull = cv2.convexHull(contours[max_idx])
    hull_points = []
    for pt in hull:
        hull_points.append(pt[0])
    return boundary_points, hull_points


def get_mask_area_img(img,pts):
    # print(pts)
    im0 = img.copy()
    mask = np.zeros(im0.shape[:2], np.uint8)
    cv2.polylines(mask, [pts], 1, 255)  # 描绘边缘
    cv2.fillPoly(mask, [pts], 255)
    dst = cv2.bitwise_and(im0, im0, mask=mask)
    return dst


def project_point_to_line(pt,line):
    k,b = line_to_params(line)
    k_ = -1.0/k
    pt_ = [pt[0]+1,pt[1]+k_]
    line_ = [pt,pt_]
    c_point = calc_cross_point(line,line_)
    return c_point


def params_to_line(k,b,w=1920):
    return [[0,b],[w,w*k+b]]


def line_to_params(line):
    p1,p2 = line
    x1,y1 = p1
    x2,y2 = p2
    if x2 - x1 == 0:  # L1 直线斜率不存在
        k = None
        b = 0
    else:
        k = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
        b = y1 * 1.0 - x1 * k * 1.0  # 整型转浮点型是关键
    return k,b


def dis_between_two_points(pt1, pt2):
    p1 = np.array(pt1)
    p2 = np.array(pt2)
    d = np.linalg.norm(p1 - p2)
    return d


def line_dis(line):
    return dis_between_two_points(line[0], line[1])


def divide_line_to_points(line, gap):
    d = line_dis(line)
    p1, p2 = line
    divide_num = math.floor(d / gap)
    if divide_num == 0:
        return line
    divide_points = []
    x_item = (p2[0] - p1[0]) / divide_num
    y_item = (p2[1] - p1[1]) / divide_num
    for i in range(divide_num + 1):
        divide_points.append([p1[0] + x_item * i, p1[1] + y_item * i])
    return divide_points


def point_distance_line(point, line):
    line_point1, line_point2 = line
    # 计算向量
    point = np.array(point)
    line_point1 = np.array(line_point1)
    line_point2 = np.array(line_point2)
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    if np.linalg.norm(line_point1 - line_point2) == 0.0:
        return 0.0
    distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
    return distance


def calc_cross_point(line1, line2):  # 计算交点函数
    x1 = line1[0][0]  # 取直线1的第一个点坐标
    y1 = line1[0][1]
    x2 = line1[1][0]  # 取直线1的第二个点坐标
    y2 = line1[1][1]

    x3 = line2[0][0]  # 取直线2的第一个点坐标
    y3 = line2[0][1]
    x4 = line2[1][0]  # 取直线2的第二个点坐标
    y4 = line2[1][1]

    if x2 - x1 == 0:  # L1 直线斜率不存在
        k1 = None
        b1 = 0
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
        b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键

    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0

    if k1 is None and k2 is None:  # L1与L2直线斜率都不存在，两条直线均与y轴平行
        if x1 == x3:  # 两条直线实际为同一直线
            return [x1, y1]  # 均为交点，返回任意一个点
        else:
            return None  # 平行线无交点
    elif k1 is not None and k2 is None:  # 若L2与y轴平行，L1为一般直线，交点横坐标为L2的x坐标
        x = x3
        y = k1 * x * 1.0 + b1 * 1.0
    elif k1 is None and k2 is not None:  # 若L1与y轴平行，L2为一般直线，交点横坐标为L1的x坐标
        x = x1
        y = k2 * x * 1.0 + b2 * 1.0
    else:  # 两条一般直线
        if k1 == k2:  # 两直线斜率相同
            if b1 == b2:  # 截距相同，说明两直线为同一直线，返回任一点
                return [x1, y1]
            else:  # 截距不同，两直线平行，无交点
                return None
        else:  # 两直线不平行，必然存在交点
            x = (b2 - b1) * 1.0 / (k1 - k2)
            y = k1 * x * 1.0 + b1 * 1.0
    return [x,y]


def get_perspective_mat(src, dst):
    src = np.array(src, np.float32)
    dst = np.array(dst, np.float32)
    p = cv2.getPerspectiveTransform(src, dst)
    return p


def point_perspective_trans(pt,perspective_mat):
    xyz = np.dot(perspective_mat, [pt[0], pt[1], 1])
    x, y, z = xyz
    return [x/z, y/z]


if __name__ == '__main__':
    # print(divide_line_to_points([[0, 0], [5, 4]], 1.1))
    # print(dis_between_two_points([0,0],[1,0]))
    print(point_distance_line([2,-1],[0,0],[0,1]))

