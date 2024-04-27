from tqdm import tqdm

from camera_calibration.xyz_trans import *
from tools.geometry import *


def init_camera_to_world_map(img_size, cal_params, ori_pt, center_point):
    w,h = img_size
    x_map = np.zeros((h,w),dtype=np.float32)
    y_map = np.zeros((h,w),dtype=np.float32)
    for x in tqdm(range(w), ncols=100):
        for y in range(h):
            pt = [x,y]
            x_,y_ = camera_to_world2_xy(pt, cal_params, ori_pt, center_point=center_point)
            x_map[y][x] = x_
            y_map[y][x] = y_
    return [x_map,y_map]


def modify_mask(mask_, boxes):
    """
    修改掩膜数组，将非零值置为1，然后将指定的包围盒区域内的值置为0。

    参数:
    mask -- 输入的掩膜数组 (numpy array)
    box -- 定义的包围盒，格式为[x1, y1, x2, y2]

    返回:
    修改后的掩膜数组
    """
    mask = mask_.copy()
    # 将mask中所有非零值置为1
    mask[mask != 0] = 1

    # 从box中提取坐标
    for x1, y1, x2, y2 in boxes:
        # 确保坐标在数组边界内
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, mask.shape[1]), min(y2, mask.shape[0])

        # 将包围盒区域内的值置为0
        mask[y1:y2, x1:x2] = 0

    return mask


def find_min_value(mask, array):
    """
    找到掩膜为1的位置对应另一个数组中的最小值。

    参数:
    mask -- 输入的二值掩膜 (numpy array)，由0和1构成
    array -- 输入的与掩膜大小相同的数组

    返回:
    掩膜为1的位置对应的数组中的最小值
    """
    # 检查输入的合法性
    assert mask.shape == array.shape, "Mask and array must have the same shape."
    assert set(np.unique(mask)).issubset({0, 1}), "Mask must be binary (contains only 0 and 1)."

    # 使用掩膜选取对应位置的元素
    masked_elements = array[mask == 1]

    # 检查是否有掩膜为1的元素
    if masked_elements.size == 0:
        return None
    # 返回掩膜为1的位置的最小值
    return np.min(masked_elements)


def expand_box_until_condition(binary_array, bbox,mode=1,img_size=None):
    img_size = [1920, 1080]
    """
    给定一个二值数组和一个包围盒，不断向上扩展包围盒，直到新增行的和小于3。

    参数:
    binary_array -- 输入的二值数组 (numpy array)
    bbox -- 初始包围盒，格式为 [x1, y1, x2, y2]

    返回:
    修改后的包围盒
    """
    x1, y1, x2, y2 = bbox

    if mode == 1:
        flag = True
        while flag:
            flag = False
            while y1 > 0:  # 确保y1不超出数组边界
                new_row_sum = np.sum(binary_array[y1 - 1, x1:x2 + 1])
                if new_row_sum < 3:
                    break
                flag = True
                y1 -= 1  # 向上扩展包围盒
            while x1 > 0:  # 确保y1不超出数组边界
                new_row_sum = np.sum(binary_array[y1:y2+1, x1-1])
                if new_row_sum < 3:
                    break
                flag = True
                x1 -= 1  # 向上扩展包围盒
            while y2 < img_size[1]-1:  # 确保y1不超出数组边界
                new_row_sum = np.sum(binary_array[y2 + 1, x1:x2 + 1])
                if new_row_sum < 3:
                    break
                flag = True
                y2 += 1  # 向上扩展包围盒
            while x2 < img_size[0]-1:  # 确保y1不超出数组边界
                new_row_sum = np.sum(binary_array[y1:y2+1, x2+1])
                if new_row_sum < 3:
                    break
                flag = True
                x2 += 1  # 向上扩展包围盒
    else:
        while y1 < y2:  # 确保y1不超出数组边界
            new_row_sum = np.sum(binary_array[y1, x1:x2 + 1])
            if new_row_sum > 0:
                break
            y1 += 1  # 向上扩展包围盒
        while y2 > y1:  # 确保y1不超出数组边界
            new_row_sum = np.sum(binary_array[y2, x1:x2 + 1])
            if new_row_sum > 0:
                break
            y2 -= 1  # 向上扩展包围盒
        while x1 < x2:  # 确保y1不超出数组边界
            new_row_sum = np.sum(binary_array[y1:y2+1, x1])
            if new_row_sum > 0:
                break
            x1 += 1  # 向上扩展包围盒
        while x2 > x1:  # 确保y1不超出数组边界
            new_row_sum = np.sum(binary_array[y1:y2+1, x2])
            if new_row_sum > 0:
                break
            x2 -= 1  # 向上扩展包围盒
    return [x1, y1, x2, y2]


def calculate_intersection(box1, box2):
    """
    计算两个包围盒的交集区域。

    参数:
    box1 -- 第一个包围盒，格式为[x1, y1, x2, y2]
    box2 -- 第二个包围盒，格式为[x1, y1, x2, y2]

    返回:
    交集的面积，如果没有交集则返回0
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right > x_left and y_bottom > y_top:
        return (x_right - x_left) * (y_bottom - y_top)
    return 0


def find_overlaps(bounding_boxes):
    """
    查找包围盒列表中所有包围盒之间的重叠关系，并返回一个字典。

    参数:
    bounding_boxes -- 包围盒列表，每个元素的格式为[x1, y1, x2, y2]

    返回:
    字典，键为包围盒索引，值为与之重叠的包围盒索引列表
    """
    overlaps = {}
    n = len(bounding_boxes)
    for i in range(n):
        overlaps[i] = []
        for j in range(n):
            if i != j:
                if calculate_intersection(bounding_boxes[i], bounding_boxes[j]) > 0:
                    overlaps[i].append(j)

    # 清理字典，去除空列表
    # overlaps = {k: v for k, v in overlaps.items() if v}

    return overlaps


def calc_box_occupy(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    # area_a = (y2 - y1) * (x2 - x1)
    # area_b = (y2_ - y1_) * (x2_ - x1_)
    occupy1 = (y2 - y1)/(y2_ - y1_)
    occupy2 = (x2 - x1)/(x2_ - x1_)
    return max(occupy1,occupy2)


def numpy_to_int_list(box):
    if all(isinstance(item, int) for item in box):
        return box
    # Convert elements to integer type
    box_int = box.astype(int)
    # Convert the numpy array to a Python list
    box_list = box_int.tolist()
    return box_list


def max_box(box,box1):
    x1,y1,x2,y2 = box
    x1_,y1_,x2_,y2_ = box1
    return [min(x1,x1_),min(y1,y1_),max(x2,x2_),max(y2,y2_)]


def fix_bounding_box(box, total_boxes, mask, img_size):
    box = numpy_to_int_list(box)
    for i in range(len(total_boxes)):
        total_boxes[i] = numpy_to_int_list(total_boxes[i])
    mask1 = modify_mask(mask, total_boxes)
    box_new = expand_box_until_condition(mask1, box, mode=1)
    box_new = expand_box_until_condition(mask, box_new, mode=2)
    x1,y1,x2,y2 = box_new
    if y2 >= img_size[1]-3 or x2 >= img_size[0]-3 or calc_box_occupy(box_new, box) >= 1.5 :
        box_new = box
    return np.array(box_new, dtype=np.float)


def remake_vehicle_seg_info(key_points, box_real, vehicle_width, vehicle_length, cal_params, ori_pt, center_point,
                            mode=1):
    kp1, kp2, kp3 = key_points
    x1, y1, x2, y2 = box_real
    if mode == 1:
        p2_ = camera_to_world2_xy(kp2, cal_params, ori_pt, center_point=center_point)
        p1_ = [p2_[0] - vehicle_width, p2_[1], 0]
        kp1_ = world2_xyz_to_camera(p1_, cal_params, ori_pt, center_point)
        y2 = max(kp1_[1], y2)
        line_bottom = [kp1_, kp2]
        box_bottom = [[x2, y2], [x2+1, y2]]
        kp1 = calc_cross_point(line_bottom, box_bottom)

        p1_ = camera_to_world2_xy(kp1, cal_params, ori_pt, center_point=center_point)
        p3_ = [p1_[0], p1_[1] + vehicle_length, 0]
        kp3_ = world2_xyz_to_camera(p3_, cal_params, ori_pt, center_point)
        x1 = min(kp3_[0], x1)
        line_left = [kp1, kp3_]
        box_left = [[x1, y1], [x1, y1 + 1]]
        kp3 = calc_cross_point(line_left, box_left)

        box_new = [x1, y1, x2, y2]
        key_points = [kp1, kp2, kp3]
    else:
        p3_ = camera_to_world2_xy(kp3, cal_params, ori_pt, center_point=center_point)
        p1_ = [p3_[0], p3_[1] - vehicle_length, 0]
        kp1_ = world2_xyz_to_camera(p1_, cal_params, ori_pt, center_point)
        y2 = max(kp1_[1], y2)
        line_left = [kp1_, kp3]
        box_bottom = [[x2, y2], [x2+1, y2]]
        kp1 = calc_cross_point(line_left, box_bottom)

        p1_ = camera_to_world2_xy(kp1, cal_params, ori_pt, center_point=center_point)
        p2_ = [p1_[0] + vehicle_width, p1_[1], 0]
        kp2_ = world2_xyz_to_camera(p2_, cal_params, ori_pt, center_point)
        x2 = max(kp2_[0],x2)
        line_bottom = [kp1, kp2_]
        box_right = [[x2, y2], [x2, y2 + 1]]
        kp2 = calc_cross_point(line_bottom, box_right)

        box_new = [x1, y1, x2, y2]
        key_points = [kp1, kp2, kp3]
    vehicle_length, vehicle_width, vehicle_pos_xy = get_vehicle_info_by_key_points(key_points, cal_params, ori_pt, center_point)
    return vehicle_length, vehicle_width, vehicle_pos_xy, box_new


def get_v_h_w_by_cls(cls,vehicle_length,vehicle_width):
    flag = False
    vehicle_length_, vehicle_width_ = vehicle_length,vehicle_width
    if cls in [0, 1]:
        min_vehicle_width = 1.3
        max_vehicle_width = 2.5
        min_vehicle_length = 3.8
        max_vehicle_length = 6
        default_car_width = 1.75
        default_van_width = 1.9
        default_vehicle_front_car = 3.8
        default_vehicle_front_van = 4.2
        if vehicle_length < min_vehicle_length or vehicle_width < min_vehicle_width \
                or vehicle_width > max_vehicle_width or vehicle_length > max_vehicle_length:
            vehicle_width_ = default_car_width
            vehicle_length_ = default_vehicle_front_car
            if cls == 1:
                vehicle_width_ = default_van_width
                vehicle_length_ = default_vehicle_front_van
            flag = True
    return flag,vehicle_length_,vehicle_width_


def get_vehicle_info_by_key_points(key_points,cal_params, ori_pt, center_point):
    kp1, kp2, kp3 = key_points
    p1 = camera_to_world2_xy(kp1, cal_params, ori_pt, center_point=center_point)
    p2 = camera_to_world2_xy(kp2, cal_params, ori_pt, center_point=center_point)
    p3 = camera_to_world2_xy(kp3, cal_params, ori_pt, center_point=center_point)
    vehicle_length, vehicle_width = round(p3[1] - p2[1], 3), round(p2[0] - p1[0], 3)
    vehicle_pos_xy = [round((p2[0] + p1[0]) / 2, 3), round(p1[1], 3)]
    return vehicle_length, vehicle_width, vehicle_pos_xy


def get_vehicle_infos(box,cls,mask, cal_params, ori_pt, center_point, pos_map):
    x1,y1,x2,y2 = numpy_to_int_list(box)
    pos_x_min = find_min_value(mask[y1:y2 + 1, x1:x2 + 1], pos_map[0][y1:y2 + 1, x1:x2 + 1])
    pos_y_min = find_min_value(mask[y1:y2 + 1, x1:x2 + 1], pos_map[1][y1:y2 + 1, x1:x2 + 1])
    if pos_x_min is None or pos_y_min is None:
        return False,[]
    p1_w2 = [pos_x_min, -10, 0]
    p2_w2 = [pos_x_min, 30, 0]
    p1_c = world2_xyz_to_camera(p1_w2, cal_params, ori_pt, center_point)
    p2_c = world2_xyz_to_camera(p2_w2, cal_params, ori_pt, center_point)
    p3_w2 = [-50, pos_y_min, 0]
    p4_w2 = [50, pos_y_min, 0]
    p3_c = world2_xyz_to_camera(p3_w2, cal_params, ori_pt, center_point)
    p4_c = world2_xyz_to_camera(p4_w2, cal_params, ori_pt, center_point)
    vehicle_line_left = [p1_c, p2_c]
    vehicle_line_bottom = [p3_c, p4_c]
    box_line_left = [[x1, y1], [x1, y1 + 1]]
    box_line_right = [[x2, y2], [x2, y2 + 1]]
    kp1 = calc_cross_point(vehicle_line_bottom, vehicle_line_left)
    kp2 = calc_cross_point(vehicle_line_bottom, box_line_right)
    kp3 = calc_cross_point(vehicle_line_left, box_line_left)
    key_points= [kp1,kp2,kp3]
    vehicle_length, vehicle_width, vehicle_pos_xy = get_vehicle_info_by_key_points(key_points, cal_params, ori_pt, center_point)
    flag, vehicle_length_,vehicle_width_ = get_v_h_w_by_cls(cls,vehicle_length,vehicle_width)
    if flag:
        vehicle_length, vehicle_width, vehicle_pos_xy, box_new = \
            remake_vehicle_seg_info(key_points, box, vehicle_width_, vehicle_length_, cal_params, ori_pt, center_point)
        # vehicle_length1, vehicle_width1, vehicle_pos_xy1, box_new1 = \
        #     remake_vehicle_seg_info(key_points, box, vehicle_width_, vehicle_length_, cal_params, ori_pt, center_point,mode=2)
        # iou1 = compute_iou(box_new,box)
        # iou2 = compute_iou(box_new1,box)
        # print(t.track_id,iou1,iou2)
        # if iou2 > iou1:
        #     vehicle_length, vehicle_width, vehicle_pos_xy, box_new =\
        #         vehicle_length1, vehicle_width1, vehicle_pos_xy1, box_new1
    else:
        box_new = box
    return True, [vehicle_length, vehicle_width, vehicle_pos_xy, box_new]


def compute_iou(boxA, boxB):
    # 确定交集框的坐标
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 计算交集的面积
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # 计算每个框的面积
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # 计算并集的面积
    unionArea = boxAArea + boxBArea - interArea

    # 计算 IoU
    iou = interArea / unionArea if unionArea > 0 else 0

    return iou
